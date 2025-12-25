
import torch
import torch.nn as nn
import math
from transformers import PreTrainedTokenizerFast

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.out_linear(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
    def forward(self, x): return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.norm_1, self.norm_2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        x = self.norm_1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.norm_2(x + self.dropout(self.ff(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x = self.n1(x + self.dropout(self.self_attn(x, x, x, trg_mask)))
        x = self.n2(self.dropout(self.cross_attn(x, e_outputs, e_outputs, src_mask)) + x)
        x = self.n3(x + self.dropout(self.ff(x)))
        return x

class ModelTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.pad_id = 0
        self.enc_emb = nn.Embedding(vocab_size, d_model)
        self.dec_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg):
        src_mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)
        trg_mask = (trg != self.pad_id).unsqueeze(1).unsqueeze(2)
        trg_mask = trg_mask & torch.tril(torch.ones(1, trg.size(1), trg.size(1), device=trg.device)).bool()
        e_out = self.pos_enc(self.enc_emb(src))
        for layer in self.encoder: e_out = layer(e_out, src_mask)
        d_out = self.pos_enc(self.dec_emb(trg))
        for layer in self.decoder: d_out = layer(d_out, e_out, src_mask, trg_mask)
        return self.out(d_out)


class Translator:
    def __init__(self, model_path, tokenizer_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, 
                                                 unk_token="<unk>", pad_token="<pad>", 
                                                 bos_token="<s>", eos_token="</s>")
        self.model = ModelTransformer(vocab_size=self.tokenizer.vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def translate(self, text, beam_size=5):
        
        src = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        
        
        beams = [(0.0, [self.tokenizer.bos_token_id])]
        for _ in range(64):
            candidates = []
            for score, seq in beams:
                if seq[-1] == self.tokenizer.eos_token_id:
                    candidates.append((score, seq))
                    continue
                out = self.model(src, torch.tensor([seq], device=self.device))
                probs = torch.log_softmax(out[:, -1, :], dim=-1)
                top_v, top_i = probs.topk(beam_size)
                for i in range(beam_size):
                    candidates.append((score + top_v[0][i].item(), seq + [top_i[0][i].item()]))
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
            if all(s[-1] == self.tokenizer.eos_token_id for _, s in beams): break
        
        
        decoded = self.tokenizer.decode(beams[0][1], skip_special_tokens=True)

        return decoded.split(".")[0].strip() + "." if "." in decoded else decoded

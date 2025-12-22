'''
Using prepared vocabs in BPE tokenize and add special tokens
'''

from bpe_tokenize import load_bpe_en, load_bpe_tr, bpe_en, bpe_tr

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

def load_vocab_en(bpe_path="bpe_en.json"):
    load_bpe_en(bpe_path)
    return bpe_en.token_to_id, bpe_en.id_to_token

def load_vocab_tr(bpe_path="bpe_tr.json"):
    load_bpe_tr(bpe_path)
    return bpe_tr.token_to_id, bpe_tr.id_to_token

def get_special_indices(token_to_id):
    try:
        pad = token_to_id[PAD_TOKEN]
        unk = token_to_id[UNK_TOKEN]
        sos = token_to_id[SOS_TOKEN]
        eos = token_to_id[EOS_TOKEN]
    except KeyError as e:
        raise ValueError(f"Special token vocab'da yok: {e}")
    return pad, unk, sos, eos

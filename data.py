'''
Converting a string dataset into PyTorch tensors using PyTorch tools.
Preparing (loading) the converted tensor data into model inputs for training, validation, and testing.
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from bpe_tokenize import tokenize_en, tokenize_tr



#%%
'''
Conversation sentence to tensor
'''


def numericalize(sentence, token_to_id, unk_idx, tokenize_fn, add_sos_eos=True):
    tokens = tokenize_fn(sentence)

    if add_sos_eos:
        tokens = ["<sos>"] + tokens + ["<eos>"]

    ids = []
    for t in tokens:
        ids.append(token_to_id.get(t, unk_idx))

    return torch.tensor(ids, dtype=torch.long)



def decode_ids(ids, id_to_token, stop_at_eos=False):
    tokens = []
    for i in ids:
        i = int(i)
        if i in id_to_token:
            tok = id_to_token[i]
        else:
            tok = "<unk>"

        if stop_at_eos and tok == "<eos>":
            break

        tokens.append(tok)

    return tokens


#%%
'''
Definitons class and fuctions for data loader
'''

class ParallelTranslationDataset(Dataset):
    def __init__(self, src_lines, trg_lines,
                 src_token_to_id, trg_token_to_id,
                 src_unk_idx, trg_unk_idx,
                 src_tokenize_fn, trg_tokenize_fn):


        if len(src_lines) != len(trg_lines):
            raise ValueError("Source and target line counts are not equal!")

       
        
        self.src_lines = src_lines
        self.trg_lines = trg_lines

        self.src_token_to_id = src_token_to_id
        self.trg_token_to_id = trg_token_to_id

        self.src_tokenize_fn = src_tokenize_fn
        self.trg_tokenize_fn = trg_tokenize_fn

        self.src_unk_idx = src_unk_idx
        self.trg_unk_idx = trg_unk_idx

        

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_sentence = self.src_lines[idx]
        trg_sentence = self.trg_lines[idx]

        src_ids = numericalize(src_sentence, self.src_token_to_id, self.src_unk_idx,
                       tokenize_fn=self.src_tokenize_fn, add_sos_eos=True)

        trg_ids = numericalize(trg_sentence, self.trg_token_to_id, self.trg_unk_idx,
                       tokenize_fn=self.trg_tokenize_fn, add_sos_eos=True)

        return src_ids, trg_ids


def make_collate_fn(src_pad_idx, trg_pad_idx, batch_first=True):
    def collate_fn(batch):
        src_list, trg_list = [], []
        for src_ids, trg_ids in batch:
            src_list.append(src_ids)
            trg_list.append(trg_ids)

        src_lens = torch.tensor([len(x) for x in src_list], dtype=torch.long)
        trg_lens = torch.tensor([len(x) for x in trg_list], dtype=torch.long)

        src = pad_sequence(src_list, batch_first=batch_first, padding_value=src_pad_idx)
        trg = pad_sequence(trg_list, batch_first=batch_first, padding_value=trg_pad_idx)


        # Padding mask for transformers 
        src_key_padding_mask = (src == src_pad_idx)
        trg_key_padding_mask = (trg == trg_pad_idx)

        return src, src_lens, trg, trg_lens, src_key_padding_mask, trg_key_padding_mask

    return collate_fn


#%%
'''
Definition of load function of tensor formats datas 
'''

def build_loaders_one_direction(
    train_src_lines, train_trg_lines,
    val_src_lines, val_trg_lines,
    test_src_lines, test_trg_lines,
    src_token_to_id, trg_token_to_id,
    src_unk_idx, trg_unk_idx,
    src_pad_idx, trg_pad_idx,
    src_tokenize_fn, trg_tokenize_fn,
    batch_size=64,
    shuffle_train=True,
    batch_first=True
):
 
    train_ds = ParallelTranslationDataset(
        train_src_lines, train_trg_lines,
        src_token_to_id, trg_token_to_id,
        src_unk_idx, trg_unk_idx,
        src_tokenize_fn=src_tokenize_fn,
        trg_tokenize_fn=trg_tokenize_fn
    )

    val_ds = ParallelTranslationDataset(
        val_src_lines, val_trg_lines,
        src_token_to_id, trg_token_to_id,
        src_unk_idx, trg_unk_idx,
        src_tokenize_fn=src_tokenize_fn,
        trg_tokenize_fn=trg_tokenize_fn
    )

    test_ds = ParallelTranslationDataset(
        test_src_lines, test_trg_lines,
        src_token_to_id, trg_token_to_id,
        src_unk_idx, trg_unk_idx,
        src_tokenize_fn=src_tokenize_fn,
        trg_tokenize_fn=trg_tokenize_fn
    )

   
    collate_fn = make_collate_fn(
        src_pad_idx, trg_pad_idx,
        batch_first=batch_first
    )

    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,         collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,         collate_fn=collate_fn)

    return train_loader, val_loader, test_loader



def build_loaders_both_directions(
    train_en_lines, train_tr_lines,
    val_en_lines, val_tr_lines,
    test_en_lines, test_tr_lines,
    en_token_to_id, en_id_to_token, en_pad_idx, en_unk_idx,
    tr_token_to_id, tr_id_to_token, tr_pad_idx, tr_unk_idx,
    batch_size=64,
    batch_first=True
):
    # EN -> TR
    train_loader_en2tr, val_loader_en2tr, test_loader_en2tr = build_loaders_one_direction(
        train_en_lines, train_tr_lines,
        val_en_lines, val_tr_lines,
        test_en_lines, test_tr_lines,
        en_token_to_id, tr_token_to_id,
        en_unk_idx, tr_unk_idx,
        en_pad_idx, tr_pad_idx,
        src_tokenize_fn=tokenize_en,
        trg_tokenize_fn=tokenize_tr,
        batch_size=batch_size,
        batch_first=batch_first
    )

    # TR -> EN 
    train_loader_tr2en, val_loader_tr2en, test_loader_tr2en = build_loaders_one_direction(
        train_tr_lines, train_en_lines,
        val_tr_lines, val_en_lines,
        test_tr_lines, test_en_lines,
        tr_token_to_id, en_token_to_id,
        tr_unk_idx, en_unk_idx,
        tr_pad_idx, en_pad_idx,
        src_tokenize_fn=tokenize_tr,
        trg_tokenize_fn=tokenize_en,
        batch_size=batch_size,
        batch_first=batch_first
    )

    return (train_loader_en2tr, val_loader_en2tr, test_loader_en2tr), \
           (train_loader_tr2en, val_loader_tr2en, test_loader_tr2en)

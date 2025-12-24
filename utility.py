from pathlib import Path
import pickle
import bpe_tokenize

def get_set():
    folder = Path('./dataset/vocabs')
    
    en_pkl = folder / 'en_words.pkl'
    tr_pkl = folder / 'tr_words.pkl'
    
    with open(en_pkl, "rb") as f:
        en_set = pickle.load(f)
    
    with open(tr_pkl, "rb") as f:
        tr_set = pickle.load(f)
    
    return en_set, tr_set

def detect_language(sentence):
    
    sentence = bpe_tokenize.normalize_text(sentence)
    
    words = sentence.split()
    en_count = 0
    tr_count = 0
    
    en_set, tr_set = get_set()
    
    for word in words:
        if word in en_set:
            en_count += 1
        if word in tr_set:
            tr_count += 1
    
    words_len = len(words)
    if en_count / words_len  > 0.5 or   tr_count / words_len > 0.5:
        if en_count > tr_count:
            return 'eng'
        else:
            return 'tur'
    else:
        return 'unk'
    
    
    


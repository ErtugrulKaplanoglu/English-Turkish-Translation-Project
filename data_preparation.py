#%% 
''' import operations, defining file paths and constants to be used. '''

import os
import csv
import random
import pickle



ENG_PATH =  "./Data_set/eng_sentences.tsv"
TUR_PATH = "./Data_set/tur_sentences.tsv"
LINKS_PATH = "./Data_set/eng-tur_links.tsv"

OUT_DIR = "./Data_set/prepared_datas"
VOCAB_DIR = "./Data_set/vocabs"

MIN_WORDS = 2
MAX_WORDS = 40
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 25

#%%
''' implementing the functions that will make the dataset ready for use. '''

def built_dicts(PATH):
    lan_dict = dict()
    
    with open(PATH, encoding='utf-8') as f:
        tsv = csv.reader(f, delimiter = '\t')
        for sid, lang, text in tsv:
            lan_dict[int(sid)] = text
    
    return lan_dict



def built_pairs(PATH, eng_dict, tur_dict):
    pairs = list()
    
    with open(PATH,  encoding='utf-8') as f:
        tsv = csv.reader(f, delimiter="\t")
        for eng_id, tur_id in tsv:
           eng_id = int(eng_id)
           tur_id = int(tur_id)
           if eng_id in eng_dict and tur_id in tur_dict:
               en = eng_dict[eng_id].strip()
               tr = tur_dict[tur_id].strip() 
               
               if MIN_WORDS <= len(en.split()) <= MAX_WORDS and MIN_WORDS <= len(tr.split()) <= MAX_WORDS:
                   pairs.append((en,tr))
                  
    return pairs


def write_parallel(pairs, prefix):

    os.makedirs(OUT_DIR, exist_ok=True)

    en_path = os.path.join(OUT_DIR, prefix + ".en")
    tr_path = os.path.join(OUT_DIR, prefix + ".tr")

    with open(en_path, "w", encoding="utf-8") as f_en, \
         open(tr_path, "w", encoding="utf-8") as f_tr:
        for en, tr in pairs:
            f_en.write(en + "\n")
            f_tr.write(tr + "\n")

def built_vocab(PATH):
    def clean_token(token):
        token = token.lower()

        cleaned_chars = []
        for ch in token:
            if ch.isalpha() or ch == "'":
                cleaned_chars.append(ch)
            
            cleaned = "".join(cleaned_chars)
            cleaned = cleaned.strip("'")
    
        if cleaned == "":
            return ""
        return cleaned
    
    words = set()
    with open(PATH, encoding='utf-8') as f:
        tsv = csv.reader(f, delimiter='\t')
        for sid,lang,text in tsv:
            for token in text.split():
                word = clean_token(token)
                if word:
                    words.add(word)
    
    return words

def save_vocabs(en_words, tr_words):
    os.makedirs(VOCAB_DIR, exist_ok=True)

    en_path = os.path.join(VOCAB_DIR, "en_words.pkl")
    tr_path = os.path.join(VOCAB_DIR, "tr_words.pkl")

    with open(en_path, "wb") as f:
        pickle.dump(en_words, f)

    with open(tr_path, "wb") as f:
        pickle.dump(tr_words, f)

#%%
''' function calls in main function'''

def main():
    
    eng_dict = built_dicts(ENG_PATH)
    tur_dict = built_dicts(TUR_PATH)
    
    pairs = built_pairs(LINKS_PATH, eng_dict, tur_dict)
    
    random.seed(SEED)
    random.shuffle(pairs)
    
    size_dataset = len(pairs)
    train_end = int(size_dataset * TRAIN_RATIO)
    val_end = int(size_dataset * (TRAIN_RATIO + VAL_RATIO))
    
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    write_parallel(train_pairs, "train")
    write_parallel(val_pairs, "val")
    write_parallel(test_pairs, "test")
             
    en_words = built_vocab(ENG_PATH)
    tr_words = built_vocab(TUR_PATH)
    save_vocabs(en_words, tr_words)
    

#%%
''' main function call with the protection of executing import. '''   
if __name__ == '__main__':
    main()
    


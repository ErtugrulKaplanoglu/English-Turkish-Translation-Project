import csv

ENG_PATH = "eng_sentences.tsv"
TUR_PATH = "tur_sentences.tsv"
LINKS_PATH = "eng-tur_links.tsv"

eng_dict = {}
with open(ENG_PATH, encoding="utf-8") as f:
    tsv = csv.reader(f, delimiter="\t")
    for sid, lang, text in tsv:
        eng_dict[int(sid)] = text

tur_dict = {}
with open(TUR_PATH, encoding="utf-8") as f:
    tsv = csv.reader(f, delimiter="\t")
    for sid, lang, text in tsv:
        tur_dict[int(sid)] = text

pairs = []  # (en_text, tr_text)

with open(LINKS_PATH, encoding="utf-8") as f:
    tsv = csv.reader(f, delimiter="\t")
    for eng_id, tur_id in tsv:
        eng_id = int(eng_id)
        tur_id = int(tur_id)
        if eng_id in eng_dict and tur_id in tur_dict:
            en = eng_dict[eng_id].strip()
            tr = tur_dict[tur_id].strip()        
          
            if len(en.split()) >= 2 and len(tr.split()) >= 2:
                pairs.append((en, tr))


clean_pairs = []
for en, tr in pairs:

    en = en.strip()
    tr = tr.strip()

    if not (2 <= len(en.split()) <= 40):
        continue
    if not (2 <= len(tr.split()) <= 40):
        continue

    if en.isnumeric() or tr.isnumeric():
        continue
    
    clean_pairs.append((en, tr))

pairs = clean_pairs

import random
random.shuffle(pairs)

n = len(pairs)
train_end = int(0.7 * n)      # %70 train
val_end   = int(0.80 * n)     # %10 validation

train_pairs = pairs[:train_end]
val_pairs   = pairs[train_end:val_end]
test_pairs  = pairs[val_end:]   #100 - 70 - 10 = %20 test

def write_parallel(pairs, prefix):
    with open(prefix + ".en", "w", encoding="utf-8") as f_en, \
         open(prefix + ".tr", "w", encoding="utf-8") as f_tr:
        for en, tr in pairs:
            f_en.write(en + "\n")
            f_tr.write(tr + "\n")

write_parallel(train_pairs, "train")
write_parallel(val_pairs, "val")
write_parallel(test_pairs, "test")



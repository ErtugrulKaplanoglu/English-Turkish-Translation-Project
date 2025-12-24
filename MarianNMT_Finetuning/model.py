import os
import torch
from transformers import MarianMTModel, MarianTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

path_en_tr = "models/en-tr"
path_tr_en = "models/tr-en"

print("Modeller yükleniyor, lütfen bekleyin...")

tokenizer_en_tr = MarianTokenizer.from_pretrained(path_en_tr)
model_en_tr = MarianMTModel.from_pretrained(path_en_tr).to(device)

tokenizer_tr_en = MarianTokenizer.from_pretrained(path_tr_en)
model_tr_en = MarianMTModel.from_pretrained(path_tr_en).to(device)

def cevir(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

while True:
    print("\n1: İngilizce -> Türkçe")
    print("2: Türkçe -> İngilizce")
    print("q: Çıkış")
    
    secim = input("Seçiminiz: ")
    
    if secim == "q":
        break
    
    if secim not in ["1", "2"]:
        print("Hatalı seçim yaptınız.")
        continue
        
    metin = input("Cümleyi girin: ")
    
    if secim == "1":
        sonuc = cevir(metin, model_en_tr, tokenizer_en_tr)
        print("Çeviri: " + sonuc)
    elif secim == "2":
        sonuc = cevir(metin, model_tr_en, tokenizer_tr_en)
        print("Çeviri: " + sonuc)

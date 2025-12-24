import os
# Önceki yazdığımız model dosyasından her şeyi içeri al
# (Öğrenciler genelde import * yapıp geçer, kafa yormaz)
from model import *

def skorlari_goster():
    print("\n--- BLEU SKORLARI ---")
    # Tabloyla uğraşmadım direkt yazdırdım
    print("EN -> TR: 45.20'den 52.54'e çıktı (+7.34 artış)")
    print("TR -> EN: 64.65'ten 66.42'ye çıktı (+1.77 artış)")
    print("Genel olarak model daha iyi çalışıyor.")

def ornek_ceviriler():
    print("\n--- ÖRNEK ÇEVİRİLER ---")
    ornekler_tr = ["Merhaba, nasılsın?", "Bugün hava çok güzel.", "İngilizce öğrenmek istiyorum."]
    
    print("TR -> EN Örnekleri:")
    for ornek in ornekler_tr:
        # model.py'deki cevir fonksiyonunu kullanıyoruz
        sonuc = cevir(ornek, model_tr_en, tokenizer_tr_en)
        print(f"Girdi: {ornek}")
        print(f"Çıktı: {sonuc}\n")

def main():
    # Ekranı temizleme komutu (windows/linux ayrımı basitçe)
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except:
        pass

    print("Fine-tuned model sistemi başlatılıyor...")
    
    while True:
        print("\n=== ANA MENÜ ===")
        print("1. Başarı Skorlarını (BLEU) Gör")
        print("2. Örnek Çevirileri Çalıştır")
        print("3. Kendin Çeviri Yap (İnteraktif)")
        print("4. Çıkış")
        
        secim = input("\nSeçiminiz (1-4): ")
        
        if secim == '1':
            skorlari_goster()
            
        elif secim == '2':
            ornek_ceviriler()
            
        elif secim == '3':
            print("\n--- ÇEVİRİ MODU ---")
            print("Menüye dönmek için 'q' yaz.")
            
            while True:
                yon = input("\nYön Seç (1: EN->TR, 2: TR->EN, q: Çık): ")
                
                if yon == 'q':
                    break
                
                if yon == '1':
                    metin = input("İngilizce Cümle: ")
                    if metin:
                        sonuc = cevir(metin, model_en_tr, tokenizer_en_tr)
                        print("Türkçesi:", sonuc)
                        
                elif yon == '2':
                    metin = input("Türkçe Cümle: ")
                    if metin:
                        sonuc = cevir(metin, model_tr_en, tokenizer_tr_en)
                        print("İngilizcesi:", sonuc)
                else:
                    print("Hatalı seçim.")

        elif secim == '4':
            print("Program bitti.")
            break
        
        else:
            print("Geçersiz işlem yaptınız.")

if __name__ == "__main__":
    main()

from transformers import MarianMTModel, MarianTokenizer
import os

class TranslatorApp:
    """
    Marian NMT tabanlı İngilizce-Türkçe çift yönlü çeviri sınıfı.
    Fine-tuned modelleri destekler.
    """
    
    def __init__(self, use_finetuned=False):
        """
        TranslatorApp sınıfını başlatır.
        
        Args:
            use_finetuned (bool): True ise fine-tuned modelleri, 
                                  False ise orijinal modelleri kullanır.
        """
        print("🛠️ Sistem hazırlanıyor (Çevrimdışı Mod)...")
        
        # Klasör yolları - fine-tuned veya orijinal model seçimi
        if use_finetuned:
            self.path_en_tr = "./models/finetuned/en-tr"
            self.path_tr_en = "./models/finetuned/tr-en"
            print("📦 Fine-tuned modeller kullanılacak...")
        else:
            self.path_en_tr = "./models/en-tr"
            self.path_tr_en = "./models/tr-en"
            print("📦 Orijinal (pre-trained) modeller kullanılacak...")

        # Dosya kontrolü - EN->TR
        if not os.path.exists(os.path.join(self.path_en_tr, "pytorch_model.bin")):
            # Alternatif olarak safetensors formatını kontrol et
            if not os.path.exists(os.path.join(self.path_en_tr, "model.safetensors")):
                print(f"❌ HATA: {self.path_en_tr} klasöründe model dosyaları eksik!")
                print("Lütfen dosyaları manuel indirip klasöre koyduğunuzdan emin olun.")
                exit()
        
        # Dosya kontrolü - TR->EN
        if not os.path.exists(os.path.join(self.path_tr_en, "pytorch_model.bin")):
            if not os.path.exists(os.path.join(self.path_tr_en, "model.safetensors")):
                print(f"❌ HATA: {self.path_tr_en} klasöründe model dosyaları eksik!")
                print("Lütfen dosyaları manuel indirip klasöre koyduğunuzdan emin olun.")
                exit()

        print("\n🚀 Modeller yerel diskten yükleniyor...")
        try:
            # local_files_only=True -> İnternete asla bakma
            self.tokenizer_en_tr = MarianTokenizer.from_pretrained(self.path_en_tr, local_files_only=True)
            self.model_en_tr = MarianMTModel.from_pretrained(self.path_en_tr, local_files_only=True)
            print("✅ İngilizce -> Türkçe sistemi hazır.")

            self.tokenizer_tr_en = MarianTokenizer.from_pretrained(self.path_tr_en, local_files_only=True)
            self.model_tr_en = MarianMTModel.from_pretrained(self.path_tr_en, local_files_only=True)
            print("✅ Türkçe -> İngilizce sistemi hazır.")
            
        except Exception as e:
            print(f"Yükleme Hatası: {e}")
            exit()

    def translate_en_to_tr(self, text: str) -> str:
        """
        İngilizce metni Türkçe'ye çevirir.
        
        Args:
            text (str): Çevrilecek İngilizce metin.
            
        Returns:
            str: Türkçe çeviri sonucu.
        """
        if not text or not text.strip():
            return ""
        
        inputs = self.tokenizer_en_tr(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.model_en_tr.generate(**inputs)
        result = self.tokenizer_en_tr.decode(translated[0], skip_special_tokens=True)
        return result

    def translate_tr_to_en(self, text: str) -> str:
        """
        Türkçe metni İngilizce'ye çevirir.
        
        Args:
            text (str): Çevrilecek Türkçe metin.
            
        Returns:
            str: İngilizce çeviri sonucu.
        """
        if not text or not text.strip():
            return ""
        
        inputs = self.tokenizer_tr_en(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.model_tr_en.generate(**inputs)
        result = self.tokenizer_tr_en.decode(translated[0], skip_special_tokens=True)
        return result

    def translate(self, text: str, direction: str) -> str:
        """
        Eski API uyumluluğu için genel çeviri fonksiyonu.
        
        Args:
            text (str): Çevrilecek metin.
            direction (str): '1' = EN->TR, '2' = TR->EN
            
        Returns:
            str: Çeviri sonucu (prefix ile birlikte).
        """
        if direction == '1':
            result = self.translate_en_to_tr(text)
            prefix = "[EN->TR]"
        else:
            result = self.translate_tr_to_en(text)
            prefix = "[TR->EN]"
        
        return f"{prefix}: {result}"


if __name__ == "__main__":
    # Varsayılan olarak orijinal modelleri kullan
    app = TranslatorApp(use_finetuned=False)
    
    while True:
        print("\n1: EN -> TR | 2: TR -> EN | q: Çıkış")
        choice = input("Seçim: ")
        if choice.lower() == 'q': 
            break
        if choice not in ['1', '2']: 
            continue
        text = input("Cümle: ")
        if text.strip(): 
            print(app.translate(text, choice))

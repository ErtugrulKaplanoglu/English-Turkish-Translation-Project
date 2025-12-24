"""
MarianNMT Çeviri Wrapper Modülü
main.py'den kolayca erişilebilen fonksiyonlar sağlar.

Kullanım:
    from marian_translator import marianNMT_en2tr, marianNMT_tr2en
    
    result = marianNMT_en2tr("Hello, how are you?")
    result = marianNMT_tr2en("Merhaba, nasılsın?")
"""

import os
import sys

# Model yolunu ayarla
# marian_translator.py içindeki yol kısmını şu şekilde güncelle

# Mevcut dosyanın olduğu dizini baz alarak tam yol oluşturur
current_dir = os.path.dirname(os.path.abspath(__file__))
model_en2tr_path = os.path.join(current_dir, "models", "finetuned", "en-tr")

# TranslatorApp'ı import et
from model import TranslatorApp

# Global translator instance (lazy loading)
_translator_finetuned = None
_translator_original = None


def _get_finetuned_translator():
    """Fine-tuned translator'ı lazy loading ile yükler."""
    global _translator_finetuned
    if _translator_finetuned is None:
        _translator_finetuned = TranslatorApp(use_finetuned=True)
    return _translator_finetuned


def _get_original_translator():
    """Orijinal translator'ı lazy loading ile yükler."""
    global _translator_original
    if _translator_original is None:
        print("\n" + "="*50)
        print(" Orijinal MarianNMT modeli yükleniyor...")
        print("="*50)
        _translator_original = TranslatorApp(use_finetuned=False)
    return _translator_original


# ============================================
# FINE-TUNED MODEL FONKSİYONLARI
# ============================================

def marianNMT_en2tr(text: str) -> str:
    """
    Fine-tuned MarianNMT ile İngilizce'den Türkçe'ye çeviri.
    
    Args:
        text (str): Çevrilecek İngilizce metin.
        
    Returns:
        str: Türkçe çeviri.
        
    Örnek:
        >>> marianNMT_en2tr("Hello, how are you?")
        "Merhaba, nasılsın?"
    """
    translator = _get_finetuned_translator()
    return translator.translate_en_to_tr(text)


def marianNMT_tr2en(text: str) -> str:
    """
    Fine-tuned MarianNMT ile Türkçe'den İngilizce'ye çeviri.
    
    Args:
        text (str): Çevrilecek Türkçe metin.
        
    Returns:
        str: İngilizce çeviri.
        
    Örnek:
        >>> marianNMT_tr2en("Merhaba, nasılsın?")
        "Hello, how are you?"
    """
    translator = _get_finetuned_translator()
    return translator.translate_tr_to_en(text)


# ============================================
# ORİJİNAL MODEL FONKSİYONLARI (Karşılaştırma için)
# ============================================

def marianNMT_en2tr_original(text: str) -> str:
    """Orijinal (fine-tune edilmemiş) model ile EN->TR çeviri."""
    translator = _get_original_translator()
    return translator.translate_en_to_tr(text)


def marianNMT_tr2en_original(text: str) -> str:
    """Orijinal (fine-tune edilmemiş) model ile TR->EN çeviri."""
    translator = _get_original_translator()
    return translator.translate_tr_to_en(text)


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MarianNMT Translator Test")
    print("="*60)
    
    # Test cümleleri
    en_test = "Hello, how are you today?"
    tr_test = "Bugün hava çok güzel."
    
    print(f"\nTest (EN→TR): '{en_test}'")
    print(f"   Fine-tuned: {marianNMT_en2tr(en_test)}")
    
    print(f"\n Test (TR→EN): '{tr_test}'")
    print(f"   Fine-tuned: {marianNMT_tr2en(tr_test)}")
    
    print("\nTest tamamlandı!")

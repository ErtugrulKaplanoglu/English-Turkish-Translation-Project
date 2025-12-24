"""
╔══════════════════════════════════════════════════════════════════╗
║         MarianNMT Fine-Tuning Demo - Proje Sunumu                ║
║         İngilizce-Türkçe Çift Yönlü Çeviri Sistemi               ║
╚══════════════════════════════════════════════════════════════════╝
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from model import TranslatorApp

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("\n" + "═" * 65)
    print("║" + "    🌍 MarianNMT Fine-Tuned Translation System".center(62) + "║")
    print("║" + "    İngilizce ↔ Türkçe Çeviri Modeli".center(62) + "║")
    print("═" * 65)

def print_bleu_scores():
    print("\n┌───────────────────────────────────────────────────────────────┐")
    print("│                  📊 BLEU SKOR SONUÇLARI                       │")
    print("├─────────────┬────────────┬────────────┬────────────────────────┤")
    print("│ Yön         │ Öncesi     │ Sonrası    │ İyileşme               │")
    print("├─────────────┼────────────┼────────────┼────────────────────────┤")
    print("│ EN → TR     │   45.20    │   52.54    │ +7.34 (+16.24%) 🔥     │")
    print("│ TR → EN     │   64.65    │   66.42    │ +1.77 (+2.74%)  ✅     │")
    print("└─────────────┴────────────┴────────────┴────────────────────────┘")

def demo_translations(translator):
    print("\n┌───────────────────────────────────────────────────────────────┐")
    print("│                    📝 ÖRNEK ÇEVİRİLER                         │")
    print("└───────────────────────────────────────────────────────────────┘")
    
    examples_en = ["Hello, how are you?", "The weather is beautiful today.", "I want to learn Turkish."]
    examples_tr = ["Merhaba, nasılsın?", "Bugün hava çok güzel.", "İngilizce öğrenmek istiyorum."]
    
    print("\n  [EN → TR]")
    for en in examples_en:
        tr = translator.translate_en_to_tr(en)
        print(f"  📥 {en}")
        print(f"  📤 {tr}\n")
    
    print("  [TR → EN]")
    for tr in examples_tr:
        en = translator.translate_tr_to_en(tr)
        print(f"  📥 {tr}")
        print(f"  📤 {en}\n")

def interactive_mode(translator):
    print("\n" + "─" * 65)
    print("                    🎯 İNTERAKTİF ÇEVİRİ MODU")
    print("─" * 65)
    print("  Önce çeviri yönünü seçin, sonra cümlenizi yazın.")
    print("  Ana menüye dönmek için 'q' yazın.")
    print("─" * 65)
    
    while True:
        print("\n  1. English → Türkçe")
        print("  2. Türkçe → English")
        print("  q. Ana menüye dön")
        
        choice = input("\n  Yön seçin (1/2/q): ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == '1':
            text = input("  📝 English: ").strip()
            if text:
                result = translator.translate_en_to_tr(text)
                print(f"  📤 Türkçe: {result}")
        elif choice == '2':
            text = input("  📝 Türkçe: ").strip()
            if text:
                result = translator.translate_tr_to_en(text)
                print(f"  📤 English: {result}")

def main():
    clear_screen()
    print_header()
    
    print("\n  ⏳ Fine-tuned model yükleniyor...")
    translator = TranslatorApp(use_finetuned=True)
    
    while True:
        print("\n" + "─" * 65)
        print("                         📋 MENÜ")
        print("─" * 65)
        print("  1. 📊 BLEU Skorlarını Göster")
        print("  2. 📝 Örnek Çevirileri Göster")
        print("  3. 🎯 İnteraktif Çeviri Modu")
        print("  4. 🚪 Çıkış")
        print("─" * 65)
        
        choice = input("\n  Seçiminiz (1-4): ").strip()
        
        if choice == '1':
            print_bleu_scores()
        elif choice == '2':
            demo_translations(translator)
        elif choice == '3':
            interactive_mode(translator)
        elif choice == '4':
            print("\n  👋 Görüşmek üzere!")
            break
        
        input("\n  [Enter'a basın...]")

if __name__ == "__main__":
    main()

import utility as util
from marian_translator import marianNMT_en2tr, marianNMT_tr2en
from transformers_model import Translator
from seq2seq_translator_model import translate_en2tr, translate_tr2en

en2tr_engine = Translator("models/base_en2tr_ep10.pt", "models/tokenizer_bpe.json")
tr2en_engine = Translator("models/final_tr2en_model.pt", "models/tokenizer_bpe.json")

while True:
    source = input('Cümlenizi Giriniz / Enter Your Sentence: ')
        
    lang = util.detect_language(source)
    
    if lang == 'eng':
        
        print(en2tr_engine.translate(source) + '\t(translated by Transformers Model.)')
        print(translate_en2tr(source) + '\t(translated by Seq2Seq Model.)')
        print(marianNMT_en2tr(source) + '\t(translated by MarianNMT.)')
        pass
       
    elif lang == 'tur':
        
        print(tr2en_engine.translate(source) + '\t(Transformer Model  ile çevrildi.)')
        print(translate_tr2en(source) + '\t(Seq2Seq Model ile çevrildi.)')
        print(marianNMT_tr2en(source) + '\t(MarianNMT ile çevrildi.)')
        pass
    
    elif lang == 'unk':
        print('Unknown language!!!')
        
    cont = input('\n Devam edilsin mi/is Continued? (y/n)').lower()
    if cont == 'n':
        break


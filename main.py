import utility as util
from marian_translator import marianNMT_en2tr, marianNMT_tr2en
from transformers_model import Translator
from seq2seq_translator_model import translate_en2tr, translate_tr2en

en2tr_engine = Translator("models/base_en2tr_ep10.pt", "models/tokenizer_bpe.json")
tr2en_engine = Translator("models/final_tr2en_model.pt", "models/tokenizer_bpe.json")

while True:
    source = input('Cümlenizi Giriniz / Enter Your Sentence: ')
    
    if source == 'exit_':
        break
    
    lang = util.detect_language(source)
    
    #dil tespitine göre çevirici modellere çağrı yapılan kısım
    if lang == 'eng':
    
        #print(en2tr_translate(source))
        print(en2tr_engine.translate(source))
        print(translate_en2tr(source))
        print(marianNMT_en2tr(source) + '\t(translated by MarianNMT.)')
        pass
       
    elif lang == 'tur':
        #print(en2tr_translate(source))
        print(tr2en_engine.translate(source))
        print(marianNMT_tr2en(source) + '\t(MarianNMT ile çevrildi.)')
        pass
    
    elif lang == 'unk':
        print('Unknown language!!!')
   

    



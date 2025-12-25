'''
This module interacts with the user. 
It first identifies the language of the sentence received from the user. 
Then, it sends the sentence as arguments to the models according to its language. 
It prints the resulting translations along with the information of the translating model.
'''

import utility as util
from marian_translator import marianNMT_en2tr, marianNMT_tr2en
from transformers_model import Translator
from seq2seq_translator_model import translate_en2tr, translate_tr2en

en2tr_engine = Translator("models/en2tr_final_model.pt", "models/tokenizer_bpe.json")
tr2en_engine = Translator("models/tr2en_final.pt", "models/tokenizer_bpe.json")

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


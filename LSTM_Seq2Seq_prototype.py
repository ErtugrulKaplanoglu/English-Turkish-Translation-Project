# Prototype Source File of own LSTM Seq2Seq Model
# In this prototype is used Tensorflow Keras Library and Numpy Library
# For the prototype, the aim at this stage was to translate only from English to Turkish.

'''The print function was used at some points in the source file to observe the prototype's output. 
These statements will be removed from the code during the construction of the main model.'''


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Dataset is preparated by using data_preparation.py module
TRAIN_EN_PATH = "train.en"
TRAIN_TR_PATH = "train.tr"


with open(TRAIN_EN_PATH, encoding="utf-8") as f:
    en_sentences = [line.strip() for line in f]

with open(TRAIN_TR_PATH, encoding="utf-8") as f:
    tr_sentences = [line.strip() for line in f]

print("Total sample size:", len(en_sentences))


# For prototype we used a small part of our preparated dataset
NUM_SAMPLES = 10000
en_sentences = en_sentences[:NUM_SAMPLES]
tr_sentences = tr_sentences[:NUM_SAMPLES]

print("Used sample size:", len(en_sentences))
print("Sample EN:", en_sentences[0])
print("Sample TR:", tr_sentences[0])

# For decoder, we added start end finish flag
tr_sentences_in  = ["<sos> " + s for s in tr_sentences]
tr_sentences_out = [s + " <eos>" for s in tr_sentences]

print("TR_in  example:", tr_sentences_in[0])
print("TR_out example:", tr_sentences_out[0])


#Tokenizer process with Keras Tokenizer 
'''In the tokenizer process, words are converted to numbers. 
This is a preliminary step in creating the Numpy matrix that will be given as input to the LSTM'''

num_words_src = 10000  
src_tokenizer = Tokenizer(num_words=num_words_src, oov_token="<unk>")
src_tokenizer.fit_on_texts(en_sentences)

en_sequences = []
for sentence in en_sentences:
    seq = src_tokenizer.texts_to_sequences([sentence])
    seq = seq[0]                                       
    en_sequences.append(seq)
    
src_vocab_size = len(src_tokenizer.word_index) + 1
print("EN vocab size:", src_vocab_size)


num_words_tgt = 10000
tgt_tokenizer = Tokenizer(num_words=num_words_tgt, oov_token="<unk>")
tgt_tokenizer.fit_on_texts(tr_sentences_in)
tgt_tokenizer.fit_on_texts(tr_sentences_out)
tr_in_sequences = []
for sentence in tr_sentences_in:
    seq = tgt_tokenizer.texts_to_sequences([sentence])
    seq = seq[0]
    tr_in_sequences.append(seq)

tr_out_sequences = []
for sentence in tr_sentences_out:
    seq = tgt_tokenizer.texts_to_sequences([sentence])
    seq = seq[0]
    tr_out_sequences.append(seq)
    
tgt_vocab_size = len(tgt_tokenizer.word_index) + 1
print("TR vocab size:", tgt_vocab_size)

#padding process
'''LSTM accepts array that fixed length as input, in padding process, Numpy matrices lengths are fixed'''

max_len_src = 40  
max_len_tgt = 40

encoder_input_data_list = []
for seq in en_sequences:
    new_seq = list(seq)
    while len(new_seq) < max_len_src:
        new_seq.append(0)

    if len(new_seq) > max_len_src:
        new_seq = new_seq[:max_len_src]

    encoder_input_data_list.append(new_seq)
encoder_input_data = np.array(encoder_input_data_list)

decoder_input_data_list = []
for seq in tr_in_sequences:
    new_seq = list(seq)

    while len(new_seq) < max_len_tgt:
        new_seq.append(0)

    if len(new_seq) > max_len_tgt:
        new_seq = new_seq[:max_len_tgt]

    decoder_input_data_list.append(new_seq)
decoder_input_data = np.array(decoder_input_data_list)

decoder_target_data_list = []
for seq in tr_out_sequences:
    new_seq = list(seq)

    while len(new_seq) < max_len_tgt:
        new_seq.append(0)

    if len(new_seq) > max_len_tgt:
        new_seq = new_seq[:max_len_tgt]

    decoder_target_data_list.append(new_seq)
decoder_target_data = np.array(decoder_target_data_list)

print("encoder_input_data shape:", encoder_input_data.shape)
print("decoder_input_data shape:", decoder_input_data.shape)
print("decoder_target_data shape:", decoder_target_data.shape)

#LSTM 

emb_dim = 256     
latent_dim = 256

# --------- Encoder ---------
encoder_inputs = Input(shape=(max_len_src,), name="encoder_inputs")

encoder_embedding_layer = Embedding(
    input_dim=src_vocab_size,  
    output_dim=emb_dim,        
    mask_zero=True,             
    name="encoder_embedding"
)

encoder_embedded = encoder_embedding_layer(encoder_inputs)

encoder_lstm_layer = LSTM(
    latent_dim,         
    return_state=True, 
    name="encoder_lstm"
)

encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm_layer(encoder_embedded)
encoder_states = [encoder_state_h, encoder_state_c]

# --------- Decoder ---------
decoder_inputs = Input(shape=(max_len_tgt,), name="decoder_inputs")

decoder_embedding_layer = Embedding(
    input_dim=tgt_vocab_size,  
    output_dim=emb_dim,       
    mask_zero=True,          
    name="decoder_embedding"
)

decoder_embedded = decoder_embedding_layer(decoder_inputs)

decoder_lstm_layer = LSTM(
    latent_dim,
    return_sequences=True,   
    return_state=True,       
    name="decoder_lstm"
)
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm_layer(
    decoder_embedded,
    initial_state=encoder_states
)

decoder_dense_layer = Dense(
    tgt_vocab_size,
    activation="softmax",
    name="decoder_output"
)
decoder_outputs = decoder_dense_layer(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()


# Model training process

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


decoder_target_data_exp = np.expand_dims(decoder_target_data, -1)

history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data_exp,
    batch_size=64,
    epochs=3,          
    validation_split=0.1
)

model.save("seq2seq_lstm_en_tr.h5")
print("Model saved: seq2seq_lstm_en_tr.h5")




 
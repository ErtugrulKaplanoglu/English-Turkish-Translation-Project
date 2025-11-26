# English–Turkish Translation Project  
Seq2Seq LSTM Prototype

This project aims to build a basic English-to-Turkish translation system using a Seq2Seq LSTM model.  
The dataset consists of cleaned parallel sentence pairs that are preprocessed and split into training, validation, and test sets.

---

## 📁 Project Files
- **data_preparation.py** – Preprocesses the dataset and creates train/val/test splits  
- **LSTM_Seq2Seq_prototype.py** – Trains the Seq2Seq LSTM model  
- **train.en / train.tr** – Training data  
- **val.en / val.tr** – Validation data  
- **test.en / test.tr** – Test data  

---

## 🧠 Model Overview
- Encoder: Embedding + LSTM  
- Decoder: Embedding + LSTM + Dense (softmax)  
- Tokenization with Keras `Tokenizer`  
- `<sos>` and `<eos>` tokens used for decoder sequences  
- Loss: Sparse Categorical Crossentropy  

---

## 🎯 Purpose
This prototype provides the foundation for building a full translation system.  
Planned next steps include:
- Implementing the inference model  
- Adding an attention mechanism  
- Evaluating with BLEU score  
- Expanding the dataset  

---

## 📌 Note
This README will be updated as the project evolves.

# 🔧 MarianNMT Fine-Tuning

This folder contains the **fine-tuning** process of Helsinki-NLP's MarianMT model on our custom dataset.

## 🎯 Purpose

The main goal of this project is to compare our LSTM model (trained from scratch) with a pre-trained model. The fine-tuned MarianMT model in this folder serves as a **baseline (reference)** for comparison.

## 📊 Fine-Tuning Results

| Translation Direction | BLEU (Before) | BLEU (After) | Improvement |
|-----------------------|---------------|--------------|-------------|
| EN → TR               | 45.20         | **52.54**    | +16.24%     |
| TR → EN               | 64.65         | **66.42**    | +2.74%      |

> A BLEU score of 50+ indicates professional translation quality.



## 🚀 Usage

### Run Demo
```bash
python demo.py
```

### Run Fine-Tuning
```bash
python finetune.py
```

## ⚙️ Training Details

- **Model:** Helsinki-NLP/opus-mt-en-tr and tr-en
- **Dataset:** 50,000 sentence pairs
- **Epochs:** 3
- **Batch Size:** 8
- **Learning Rate:** 2e-5

## 📥 Fine-Tuned Model Download

Model files exceed GitHub's size limit (300MB+) and are shared separately.

> Place model files in the `models/finetuned/` directory.

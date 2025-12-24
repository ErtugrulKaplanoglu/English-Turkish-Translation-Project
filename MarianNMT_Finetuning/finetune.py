"""
Marian NMT Fine-Tuning Script
İngilizce-Türkçe Çift Yönlü Çeviri Modeli İnce Ayarı

Bu script, önceden eğitilmiş Marian NMT modellerini kendi veri setinizle
ince ayar yaparak daha iyi performans elde etmenizi sağlar.

Özellikler:
- BLEU skoru hesaplama (eğitim öncesi ve sonrası karşılaştırma)
- Detaylı loglama sistemi (dosya + konsol)
- Eğitim metrikleri raporlama
- JSON formatında sonuç kaydetme

Kullanım:
    python finetune.py

Gerekli Kütüphaneler:
    pip install transformers datasets torch sentencepiece sacremoses evaluate
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
from transformers import (
    MarianMTModel, 
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from datasets import Dataset
import evaluate
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ============================================
# LOGLAMA SİSTEMİ (Logging System)
# ============================================

class Logger:
    """
    Çift çıkışlı loglama sistemi: Hem konsol hem dosya.
    Ders projesi için tüm eğitim sürecini kaydeder.
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Logger'ı başlatır.
        
        Args:
            log_dir: Log dosyalarının kaydedileceği klasör
            experiment_name: Deney adı (dosya isimlendirmede kullanılır)
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log klasörünü oluştur
        os.makedirs(log_dir, exist_ok=True)
        
        # Log dosyası yolu
        self.log_file = os.path.join(
            log_dir, 
            f"{experiment_name}_{self.timestamp}.log"
        )
        
        # Metrikler için JSON dosyası
        self.metrics_file = os.path.join(
            log_dir,
            f"{experiment_name}_{self.timestamp}_metrics.json"
        )
        
        # Logger'ı yapılandır
        self._setup_logger()
        
        # Metrikleri saklamak için
        self.metrics_history = {
            "experiment_name": experiment_name,
            "timestamp": self.timestamp,
            "config": {},
            "training_logs": [],
            "evaluation_results": {},
            "bleu_scores": {
                "before_finetuning": {},
                "after_finetuning": {}
            }
        }
    
    def _setup_logger(self):
        """Logger'ı yapılandırır."""
        # Root logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Mevcut handler'ları temizle
        self.logger.handlers = []
        
        # Dosya handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        # Konsol handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Bilgi mesajı loglar."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Debug mesajı loglar (sadece dosyaya)."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Uyarı mesajı loglar."""
        self.logger.warning(f"⚠️ {message}")
    
    def error(self, message: str):
        """Hata mesajı loglar."""
        self.logger.error(f"❌ {message}")
    
    def section(self, title: str):
        """Bölüm başlığı loglar."""
        separator = "=" * 60
        self.logger.info(f"\n{separator}")
        self.logger.info(f"  {title}")
        self.logger.info(f"{separator}")
    
    def log_config(self, config: dict):
        """Yapılandırmayı loglar."""
        self.metrics_history["config"] = config
        self.debug(f"Yapılandırma: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    def log_training_step(self, step: int, loss: float, learning_rate: float = None):
        """Eğitim adımını loglar."""
        log_entry = {
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics_history["training_logs"].append(log_entry)
        self.debug(f"Step {step}: loss={loss:.4f}, lr={learning_rate}")
    
    def log_bleu_score(self, phase: str, direction: str, score: float, details: dict = None):
        """
        BLEU skorunu loglar.
        
        Args:
            phase: 'before_finetuning' veya 'after_finetuning'
            direction: 'en_to_tr' veya 'tr_to_en'
            score: BLEU skoru
            details: Ek detaylar (precision, length ratio vb.)
        """
        self.metrics_history["bleu_scores"][phase][direction] = {
            "score": score,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.info(f"   📊 BLEU Skoru ({direction}): {score:.4f}")
    
    def log_evaluation(self, direction: str, results: dict):
        """Değerlendirme sonuçlarını loglar."""
        self.metrics_history["evaluation_results"][direction] = results
    
    def save_metrics(self):
        """Metrikleri JSON dosyasına kaydeder."""
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        self.info(f"📁 Metrikler kaydedildi: {self.metrics_file}")
    
    def generate_report(self) -> str:
        """
        Eğitim raporunu oluşturur.
        
        Returns:
            str: Markdown formatında rapor
        """
        report = []
        report.append("# Fine-Tuning Eğitim Raporu")
        report.append(f"\n**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Deney Adı:** {self.experiment_name}")
        
        # Yapılandırma
        report.append("\n## Eğitim Yapılandırması")
        config = self.metrics_history.get("config", {})
        for key, value in config.items():
            report.append(f"- **{key}:** {value}")
        
        # BLEU Skorları Karşılaştırması
        report.append("\n## BLEU Skor Karşılaştırması")
        report.append("\n| Yön | Fine-Tuning Öncesi | Fine-Tuning Sonrası | İyileşme |")
        report.append("|-----|-------------------|--------------------|---------| ")
        
        before = self.metrics_history["bleu_scores"]["before_finetuning"]
        after = self.metrics_history["bleu_scores"]["after_finetuning"]
        
        for direction in ["en_to_tr", "tr_to_en"]:
            before_score = before.get(direction, {}).get("score", 0)
            after_score = after.get(direction, {}).get("score", 0)
            improvement = after_score - before_score
            improvement_pct = (improvement / before_score * 100) if before_score > 0 else 0
            
            direction_label = "EN → TR" if direction == "en_to_tr" else "TR → EN"
            report.append(
                f"| {direction_label} | {before_score:.4f} | {after_score:.4f} | "
                f"{'+' if improvement >= 0 else ''}{improvement:.4f} ({improvement_pct:+.2f}%) |"
            )
        
        # Değerlendirme Sonuçları
        report.append("\n## Değerlendirme Sonuçları")
        for direction, results in self.metrics_history["evaluation_results"].items():
            report.append(f"\n### {direction}")
            for metric, value in results.items():
                report.append(f"- **{metric}:** {value}")
        
        return "\n".join(report)
    
    def save_report(self, output_path: str):
        """Raporu Markdown dosyasına kaydeder."""
        report = self.generate_report()
        report_file = os.path.join(output_path, f"training_report_{self.timestamp}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        self.info(f"📄 Rapor kaydedildi: {report_file}")
        return report_file


# ============================================
# BLEU SKOR HESAPLAMA (BLEU Score Calculator)
# ============================================

class BLEUCalculator:
    """
    BLEU skoru hesaplama sınıfı.
    sacrebleu kütüphanesini kullanır.
    """
    
    def __init__(self):
        """BLEU metriğini yükler."""
        self.bleu_metric = evaluate.load("sacrebleu")
    
    def calculate_bleu(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Tuple[float, dict]:
        """
        BLEU skorunu hesaplar.
        
        Args:
            predictions: Model çıktıları
            references: Referans çeviriler
            
        Returns:
            Tuple[float, dict]: BLEU skoru ve detaylı sonuçlar
        """
        # sacrebleu referansları liste içinde liste olarak bekler
        references_wrapped = [[ref] for ref in references]
        
        results = self.bleu_metric.compute(
            predictions=predictions,
            references=references_wrapped
        )
        
        details = {
            "score": results["score"],
            "counts": results.get("counts", []),
            "totals": results.get("totals", []),
            "precisions": results.get("precisions", []),
            "bp": results.get("bp", 0),  # Brevity penalty
            "sys_len": results.get("sys_len", 0),
            "ref_len": results.get("ref_len", 0)
        }
        
        return results["score"], details
    
    def evaluate_model(
        self,
        model: MarianMTModel,
        tokenizer: MarianTokenizer,
        test_sources: List[str],
        test_references: List[str],
        batch_size: int = 32,
        max_length: int = 128
    ) -> Tuple[float, dict, List[str]]:
        """
        Modeli değerlendirir ve BLEU skoru hesaplar.
        
        Args:
            model: Değerlendirilecek model
            tokenizer: Tokenizer
            test_sources: Kaynak cümleler
            test_references: Referans çeviriler
            batch_size: Batch boyutu
            max_length: Maksimum çıktı uzunluğu
            
        Returns:
            Tuple: BLEU skoru, detaylar ve çeviriler
        """
        model.eval()
        device = next(model.parameters()).device
        predictions = []
        
        # Batch'ler halinde çeviri yap
        for i in range(0, len(test_sources), batch_size):
            batch = test_sources[i:i + batch_size]
            
            inputs = tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_length
            ).to(device)
            
            with torch.no_grad():
                translated = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
            predictions.extend(decoded)
        
        # BLEU hesapla
        bleu_score, details = self.calculate_bleu(predictions, test_references)
        
        return bleu_score, details, predictions


# ============================================
# TRAINER CALLBACK (Eğitim Sırasında Loglama)
# ============================================

class LoggingCallback(TrainerCallback):
    """Eğitim sırasında metrikleri loglayan callback."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.current_step = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Her log adımında çağrılır."""
        if logs:
            loss = logs.get("loss", logs.get("eval_loss", None))
            lr = logs.get("learning_rate", None)
            
            if loss is not None:
                self.logger.log_training_step(state.global_step, loss, lr)
                
                if state.global_step % 100 == 0:
                    self.logger.info(
                        f"   Step {state.global_step}: loss={loss:.4f}"
                        + (f", lr={lr:.2e}" if lr else "")
                    )
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Her epoch sonunda çağrılır."""
        self.logger.info(f"   ✅ Epoch {state.epoch:.0f} tamamlandı.")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Değerlendirme sonrasında çağrılır."""
        if metrics:
            eval_loss = metrics.get("eval_loss", None)
            if eval_loss:
                self.logger.info(f"   📈 Değerlendirme loss: {eval_loss:.4f}")


# ============================================
# YAPILANDIRMA (Configuration)
# ============================================

class Config:
    """Fine-tuning ayarları"""
    
    # Veri seti yolları
    DATA_DIR = "../Data_set/prepared_datas"
    TRAIN_EN = os.path.join(DATA_DIR, "train.en")
    TRAIN_TR = os.path.join(DATA_DIR, "train.tr")
    VAL_EN = os.path.join(DATA_DIR, "val.en")
    VAL_TR = os.path.join(DATA_DIR, "val.tr")
    TEST_EN = os.path.join(DATA_DIR, "test.en")
    TEST_TR = os.path.join(DATA_DIR, "test.tr")
    
    # Model yolları
    MODEL_EN_TR = "./models/en-tr"
    MODEL_TR_EN = "./models/tr-en"
    
    # Fine-tuned model kayıt yolları
    OUTPUT_EN_TR = "./models/finetuned/en-tr"
    OUTPUT_TR_EN = "./models/finetuned/tr-en"
    
    # Log klasörü
    LOG_DIR = "./logs"
    
    # Eğitim parametreleri
    MAX_SAMPLES = 50000          # Kullanılacak maksimum örnek sayısı (None = hepsi)
    TEST_SAMPLES = 1000          # BLEU hesaplama için test örnek sayısı
    MAX_LENGTH = 128             # Maksimum token uzunluğu
    BATCH_SIZE = 8               # Batch boyutu (GPU belleğine göre ayarlayın)
    LEARNING_RATE = 2e-5         # Öğrenme oranı
    NUM_EPOCHS = 3               # Eğitim dönem sayısı
    WARMUP_STEPS = 500           # Isınma adımları
    WEIGHT_DECAY = 0.01          # Ağırlık çürümesi
    SAVE_STEPS = 1000            # Checkpoint kaydetme sıklığı
    EVAL_STEPS = 500             # Değerlendirme sıklığı
    LOGGING_STEPS = 100          # Log yazma sıklığı
    
    # Cihaz ayarları
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16 = torch.cuda.is_available()  # GPU varsa mixed precision kullan
    
    def to_dict(self) -> dict:
        """Yapılandırmayı sözlük olarak döndürür."""
        return {
            "max_samples": self.MAX_SAMPLES,
            "test_samples": self.TEST_SAMPLES,
            "max_length": self.MAX_LENGTH,
            "batch_size": self.BATCH_SIZE,
            "learning_rate": self.LEARNING_RATE,
            "num_epochs": self.NUM_EPOCHS,
            "warmup_steps": self.WARMUP_STEPS,
            "weight_decay": self.WEIGHT_DECAY,
            "device": self.DEVICE,
            "fp16": self.FP16
        }


# ============================================
# VERİ YÜKLEME
# ============================================

def load_parallel_data(
    src_file: str, 
    tgt_file: str, 
    max_samples: int = None,
    logger: Logger = None
) -> Dataset:
    """
    Paralel veri setini yükler.
    
    Args:
        src_file: Kaynak dil dosyası
        tgt_file: Hedef dil dosyası
        max_samples: Maksimum örnek sayısı
        logger: Logger nesnesi
        
    Returns:
        Dataset: HuggingFace Dataset nesnesi
    """
    msg = f"📂 Veri yükleniyor: {os.path.basename(src_file)} -> {os.path.basename(tgt_file)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f.readlines()]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f.readlines()]
    
    # Satır sayısı kontrolü
    assert len(src_lines) == len(tgt_lines), \
        f"Kaynak ve hedef dosya satır sayıları eşit olmalı! ({len(src_lines)} != {len(tgt_lines)})"
    
    # Maksimum örnek sınırlaması
    if max_samples and max_samples < len(src_lines):
        src_lines = src_lines[:max_samples]
        tgt_lines = tgt_lines[:max_samples]
    
    msg = f"   ✅ {len(src_lines)} cümle çifti yüklendi."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    return Dataset.from_dict({
        "source": src_lines,
        "target": tgt_lines
    })


def load_test_data(src_file: str, tgt_file: str, max_samples: int = 1000) -> Tuple[List[str], List[str]]:
    """Test verisini liste olarak yükler."""
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f.readlines()[:max_samples]]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f.readlines()[:max_samples]]
    
    return src_lines, tgt_lines


def preprocess_function(examples, tokenizer, max_length=128):
    """Veriyi tokenize eder."""
    inputs = examples["source"]
    targets = examples["target"]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=max_length, 
        truncation=True, 
        padding="max_length"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=max_length, 
            truncation=True, 
            padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ============================================
# FINE-TUNING FONKSİYONU
# ============================================

def finetune_model(
    model_path: str,
    output_path: str,
    train_src: str,
    train_tgt: str,
    val_src: str,
    val_tgt: str,
    test_src: str,
    test_tgt: str,
    direction: str,
    config: Config,
    logger: Logger,
    bleu_calculator: BLEUCalculator
) -> dict:
    """
    Modeli ince ayar yapar ve değerlendirir.
    
    Returns:
        dict: Eğitim sonuçları
    """
    direction_key = "en_to_tr" if "EN" in direction.split("->")[0] else "tr_to_en"
    
    logger.section(f"{direction} Modeli Fine-Tuning")
    
    # Model ve tokenizer yükle
    logger.info(f"📦 Model yükleniyor: {model_path}")
    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_path, local_files_only=True)
    model = model.to(config.DEVICE)
    logger.info(f"   ✅ Model {config.DEVICE} üzerinde yüklendi.")
    
    # Test verisini yükle
    logger.info(f"\n📊 Fine-Tuning ÖNCESİ BLEU skoru hesaplanıyor...")
    test_sources, test_references = load_test_data(test_src, test_tgt, config.TEST_SAMPLES)
    
    # Fine-tuning ÖNCE BLEU hesapla
    before_bleu, before_details, before_translations = bleu_calculator.evaluate_model(
        model, tokenizer, test_sources, test_references
    )
    logger.log_bleu_score("before_finetuning", direction_key, before_bleu, before_details)
    
    # Örnek çeviriler göster
    logger.info("\n   📝 Örnek çeviriler (fine-tuning öncesi):")
    for i in range(min(3, len(test_sources))):
        logger.debug(f"      Kaynak: {test_sources[i]}")
        logger.debug(f"      Çeviri: {before_translations[i]}")
        logger.debug(f"      Referans: {test_references[i]}")
        logger.debug("      ---")
    
    # Eğitim verisini yükle
    train_dataset = load_parallel_data(train_src, train_tgt, config.MAX_SAMPLES, logger)
    val_dataset = load_parallel_data(val_src, val_tgt, min(5000, config.MAX_SAMPLES or 5000), logger)
    
    # Veriyi tokenize et
    logger.info("\n🔄 Veri tokenize ediliyor...")
    
    def tokenize_fn(examples):
        return preprocess_function(examples, tokenizer, config.MAX_LENGTH)
    
    train_dataset = train_dataset.map(
        tokenize_fn, 
        batched=True, 
        remove_columns=["source", "target"],
        desc="Eğitim verisi"
    )
    
    val_dataset = val_dataset.map(
        tokenize_fn, 
        batched=True, 
        remove_columns=["source", "target"],
        desc="Doğrulama verisi"
    )
    
    logger.info("   ✅ Tokenizasyon tamamlandı.")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Eğitim argümanları
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=os.path.join(output_path, "logs"),
        logging_steps=config.LOGGING_STEPS,
        fp16=config.FP16,
        dataloader_num_workers=0,
        predict_with_generate=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Harici raporlamayı devre dışı bırak
    )
    
    # Trainer oluştur
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoggingCallback(logger)]
    )
    
    # Eğitimi başlat
    logger.info(f"\n🏋️ Eğitim başlıyor...")
    logger.info(f"   - Epoch sayısı: {config.NUM_EPOCHS}")
    logger.info(f"   - Eğitim örneği: {len(train_dataset)}")
    logger.info(f"   - Batch boyutu: {config.BATCH_SIZE}")
    logger.info(f"   - Öğrenme oranı: {config.LEARNING_RATE}")
    
    train_result = trainer.train()
    
    # Eğitim istatistikleri
    logger.info(f"\n📊 Eğitim tamamlandı!")
    logger.info(f"   - Toplam adım: {train_result.global_step}")
    logger.info(f"   - Eğitim süresi: {train_result.metrics.get('train_runtime', 0):.2f} saniye")
    logger.info(f"   - Son loss: {train_result.metrics.get('train_loss', 0):.4f}")
    
    # Modeli kaydet
    logger.info(f"\n💾 Model kaydediliyor: {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Fine-tuning SONRA BLEU hesapla
    logger.info(f"\n📊 Fine-Tuning SONRASI BLEU skoru hesaplanıyor...")
    
    # Fine-tuned modeli yükle
    model_finetuned = MarianMTModel.from_pretrained(output_path, local_files_only=True)
    model_finetuned = model_finetuned.to(config.DEVICE)
    
    after_bleu, after_details, after_translations = bleu_calculator.evaluate_model(
        model_finetuned, tokenizer, test_sources, test_references
    )
    logger.log_bleu_score("after_finetuning", direction_key, after_bleu, after_details)
    
    # Örnek çeviriler göster
    logger.info("\n   📝 Örnek çeviriler (fine-tuning sonrası):")
    for i in range(min(3, len(test_sources))):
        logger.info(f"      Kaynak: {test_sources[i]}")
        logger.info(f"      Çeviri: {after_translations[i]}")
        logger.info(f"      Referans: {test_references[i]}")
        logger.info("      ---")
    
    # İyileşme hesapla
    improvement = after_bleu - before_bleu
    improvement_pct = (improvement / before_bleu * 100) if before_bleu > 0 else 0
    
    logger.info(f"\n🎯 BLEU Skor Karşılaştırması ({direction}):")
    logger.info(f"   - Öncesi: {before_bleu:.4f}")
    logger.info(f"   - Sonrası: {after_bleu:.4f}")
    logger.info(f"   - İyileşme: {'+' if improvement >= 0 else ''}{improvement:.4f} ({improvement_pct:+.2f}%)")
    
    # Sonuçları logla
    results = {
        "direction": direction,
        "before_bleu": before_bleu,
        "after_bleu": after_bleu,
        "improvement": improvement,
        "improvement_percent": improvement_pct,
        "train_loss": train_result.metrics.get('train_loss', 0),
        "train_runtime": train_result.metrics.get('train_runtime', 0),
        "train_samples": len(train_dataset),
        "total_steps": train_result.global_step
    }
    logger.log_evaluation(direction_key, results)
    
    return results


# ============================================
# ANA FONKSİYON
# ============================================

def main():
    """Ana fonksiyon"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              MARIAN NMT FINE-TUNING                          ║
    ║          İngilizce-Türkçe Çift Yönlü Çeviri                  ║
    ║                                                              ║
    ║  📊 BLEU Skoru Hesaplama                                     ║
    ║  📝 Detaylı Loglama                                          ║
    ║  📄 Otomatik Rapor Oluşturma                                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    config = Config()
    
    # Logger oluştur
    logger = Logger(config.LOG_DIR, "marian_finetuning")
    logger.log_config(config.to_dict())
    
    # BLEU hesaplayıcı
    bleu_calculator = BLEUCalculator()
    
    # GPU kontrolü
    logger.section("Sistem Bilgileri")
    logger.info(f"🖥️ Cihaz: {config.DEVICE}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"   GPU: {gpu_name}")
        logger.info(f"   Bellek: {gpu_memory:.2f} GB")
    else:
        logger.warning("GPU bulunamadı. CPU ile eğitim yapılacak (daha yavaş).")
    
    logger.info(f"📁 Log dosyası: {logger.log_file}")
    logger.info(f"📊 Metrik dosyası: {logger.metrics_file}")
    
    # Çıktı klasörlerini oluştur
    os.makedirs(config.OUTPUT_EN_TR, exist_ok=True)
    os.makedirs(config.OUTPUT_TR_EN, exist_ok=True)
    
    results = {}
    
    # ==========================================
    # 1. İngilizce -> Türkçe Fine-Tuning
    # ==========================================
    results["en_to_tr"] = finetune_model(
        model_path=config.MODEL_EN_TR,
        output_path=config.OUTPUT_EN_TR,
        train_src=config.TRAIN_EN,
        train_tgt=config.TRAIN_TR,
        val_src=config.VAL_EN,
        val_tgt=config.VAL_TR,
        test_src=config.TEST_EN,
        test_tgt=config.TEST_TR,
        direction="EN -> TR",
        config=config,
        logger=logger,
        bleu_calculator=bleu_calculator
    )
    
    # ==========================================
    # 2. Türkçe -> İngilizce Fine-Tuning
    # ==========================================
    results["tr_to_en"] = finetune_model(
        model_path=config.MODEL_TR_EN,
        output_path=config.OUTPUT_TR_EN,
        train_src=config.TRAIN_TR,
        train_tgt=config.TRAIN_EN,
        val_src=config.VAL_TR,
        val_tgt=config.VAL_EN,
        test_src=config.TEST_TR,
        test_tgt=config.TEST_EN,
        direction="TR -> EN",
        config=config,
        logger=logger,
        bleu_calculator=bleu_calculator
    )
    
    # ==========================================
    # 3. Final Rapor
    # ==========================================
    logger.section("SONUÇ ÖZETİ")
    
    logger.info("\n📊 BLEU Skor Tablosu:")
    logger.info("-" * 60)
    logger.info(f"{'Yön':<15} {'Öncesi':<12} {'Sonrası':<12} {'İyileşme':<15}")
    logger.info("-" * 60)
    
    for key, res in results.items():
        direction = res["direction"]
        logger.info(
            f"{direction:<15} {res['before_bleu']:<12.4f} {res['after_bleu']:<12.4f} "
            f"{res['improvement']:+.4f} ({res['improvement_percent']:+.2f}%)"
        )
    
    logger.info("-" * 60)
    
    # Metrikleri kaydet
    logger.save_metrics()
    
    # Rapor oluştur ve kaydet
    report_file = logger.save_report(config.LOG_DIR)
    
    logger.section("TAMAMLANDI")
    logger.info("✅ Fine-tuning işlemi başarıyla tamamlandı!")
    logger.info(f"\n📁 Çıktı Dosyaları:")
    logger.info(f"   - EN->TR Model: {config.OUTPUT_EN_TR}")
    logger.info(f"   - TR->EN Model: {config.OUTPUT_TR_EN}")
    logger.info(f"   - Log dosyası: {logger.log_file}")
    logger.info(f"   - Metrikler: {logger.metrics_file}")
    logger.info(f"   - Rapor: {report_file}")
    
    return results


if __name__ == "__main__":
    main()

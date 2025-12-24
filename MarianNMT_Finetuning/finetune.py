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


class Logger:
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{self.timestamp}.log")
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_{self.timestamp}_metrics.json")
        
        self._setup_logger()
        
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
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def section(self, title: str):
        separator = "=" * 60
        self.logger.info(f"\n{separator}")
        self.logger.info(f"  {title}")
        self.logger.info(f"{separator}")
    
    def log_config(self, config: dict):
        self.metrics_history["config"] = config
        self.debug(f"Config: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    def log_training_step(self, step: int, loss: float, learning_rate: float = None):
        log_entry = {
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics_history["training_logs"].append(log_entry)
        self.debug(f"Step {step}: loss={loss:.4f}, lr={learning_rate}")
    
    def log_bleu_score(self, phase: str, direction: str, score: float, details: dict = None):
        self.metrics_history["bleu_scores"][phase][direction] = {
            "score": score,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.info(f"   BLEU Score ({direction}): {score:.4f}")
    
    def log_evaluation(self, direction: str, results: dict):
        self.metrics_history["evaluation_results"][direction] = results
    
    def save_metrics(self):
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        self.info(f"Metrics saved: {self.metrics_file}")
    
    def generate_report(self) -> str:
        report = []
        report.append("# Fine-Tuning Training Report")
        report.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Experiment:** {self.experiment_name}")
        
        report.append("\n## Training Configuration")
        config = self.metrics_history.get("config", {})
        for key, value in config.items():
            report.append(f"- **{key}:** {value}")
        
        report.append("\n## BLEU Score Comparison")
        report.append("\n| Direction | Before | After | Improvement |")
        report.append("|-----------|--------|-------|-------------|")
        
        before = self.metrics_history["bleu_scores"]["before_finetuning"]
        after = self.metrics_history["bleu_scores"]["after_finetuning"]
        
        for direction in ["en_to_tr", "tr_to_en"]:
            before_score = before.get(direction, {}).get("score", 0)
            after_score = after.get(direction, {}).get("score", 0)
            improvement = after_score - before_score
            improvement_pct = (improvement / before_score * 100) if before_score > 0 else 0
            
            direction_label = "EN -> TR" if direction == "en_to_tr" else "TR -> EN"
            report.append(f"| {direction_label} | {before_score:.4f} | {after_score:.4f} | {'+' if improvement >= 0 else ''}{improvement:.4f} ({improvement_pct:+.2f}%) |")
        
        report.append("\n## Evaluation Results")
        for direction, results in self.metrics_history["evaluation_results"].items():
            report.append(f"\n### {direction}")
            for metric, value in results.items():
                report.append(f"- **{metric}:** {value}")
        
        return "\n".join(report)
    
    def save_report(self, output_path: str):
        report = self.generate_report()
        report_file = os.path.join(output_path, f"training_report_{self.timestamp}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        self.info(f"Report saved: {report_file}")
        return report_file


class BLEUCalculator:
    def __init__(self):
        self.bleu_metric = evaluate.load("sacrebleu")
    
    def calculate_bleu(self, predictions: List[str], references: List[str]) -> Tuple[float, dict]:
        references_wrapped = [[ref] for ref in references]
        
        results = self.bleu_metric.compute(predictions=predictions, references=references_wrapped)
        
        details = {
            "score": results["score"],
            "counts": results.get("counts", []),
            "totals": results.get("totals", []),
            "precisions": results.get("precisions", []),
            "bp": results.get("bp", 0),
            "sys_len": results.get("sys_len", 0),
            "ref_len": results.get("ref_len", 0)
        }
        
        return results["score"], details
    
    def evaluate_model(self, model, tokenizer, test_sources: List[str], test_references: List[str], batch_size: int = 32, max_length: int = 128) -> Tuple[float, dict, List[str]]:
        model.eval()
        device = next(model.parameters()).device
        predictions = []
        
        for i in range(0, len(test_sources), batch_size):
            batch = test_sources[i:i + batch_size]
            
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
            
            decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
            predictions.extend(decoded)
        
        bleu_score, details = self.calculate_bleu(predictions, test_references)
        
        return bleu_score, details, predictions


class LoggingCallback(TrainerCallback):
    def __init__(self, logger: Logger):
        self.logger = logger
        self.current_step = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss", logs.get("eval_loss", None))
            lr = logs.get("learning_rate", None)
            
            if loss is not None:
                self.logger.log_training_step(state.global_step, loss, lr)
                
                if state.global_step % 100 == 0:
                    self.logger.info(f"   Step {state.global_step}: loss={loss:.4f}" + (f", lr={lr:.2e}" if lr else ""))
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self.logger.info(f"   Epoch {state.epoch:.0f} completed.")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss", None)
            if eval_loss:
                self.logger.info(f"   Eval loss: {eval_loss:.4f}")


class Config:
    DATA_DIR = "../Data_set/prepared_datas"
    TRAIN_EN = os.path.join(DATA_DIR, "train.en")
    TRAIN_TR = os.path.join(DATA_DIR, "train.tr")
    VAL_EN = os.path.join(DATA_DIR, "val.en")
    VAL_TR = os.path.join(DATA_DIR, "val.tr")
    TEST_EN = os.path.join(DATA_DIR, "test.en")
    TEST_TR = os.path.join(DATA_DIR, "test.tr")
    
    MODEL_EN_TR = "./models/en-tr"
    MODEL_TR_EN = "./models/tr-en"
    
    OUTPUT_EN_TR = "./models/finetuned/en-tr"
    OUTPUT_TR_EN = "./models/finetuned/tr-en"
    
    LOG_DIR = "./logs"
    
    MAX_SAMPLES = 50000
    TEST_SAMPLES = 1000
    MAX_LENGTH = 128
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    SAVE_STEPS = 1000
    EVAL_STEPS = 500
    LOGGING_STEPS = 100
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16 = torch.cuda.is_available()
    
    def to_dict(self) -> dict:
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


def load_parallel_data(src_file: str, tgt_file: str, max_samples: int = None, logger: Logger = None) -> Dataset:
    if logger:
        logger.info(f"Loading data: {os.path.basename(src_file)} -> {os.path.basename(tgt_file)}")
    
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f.readlines()]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f.readlines()]
    
    assert len(src_lines) == len(tgt_lines), f"Line count mismatch! ({len(src_lines)} != {len(tgt_lines)})"
    
    if max_samples and max_samples < len(src_lines):
        src_lines = src_lines[:max_samples]
        tgt_lines = tgt_lines[:max_samples]
    
    if logger:
        logger.info(f"   {len(src_lines)} sentence pairs loaded.")
    
    return Dataset.from_dict({"source": src_lines, "target": tgt_lines})


def load_test_data(src_file: str, tgt_file: str, max_samples: int = 1000) -> Tuple[List[str], List[str]]:
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f.readlines()[:max_samples]]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f.readlines()[:max_samples]]
    
    return src_lines, tgt_lines


def preprocess_function(examples, tokenizer, max_length=128):
    inputs = examples["source"]
    targets = examples["target"]
    
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def finetune_model(model_path: str, output_path: str, train_src: str, train_tgt: str, val_src: str, val_tgt: str, test_src: str, test_tgt: str, direction: str, config: Config, logger: Logger, bleu_calculator: BLEUCalculator) -> dict:
    direction_key = "en_to_tr" if "EN" in direction.split("->")[0] else "tr_to_en"
    
    logger.section(f"{direction} Model Fine-Tuning")
    
    logger.info(f"Loading model: {model_path}")
    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_path, local_files_only=True)
    model = model.to(config.DEVICE)
    logger.info(f"   Model loaded on {config.DEVICE}")
    
    logger.info(f"\nCalculating BLEU score BEFORE fine-tuning...")
    test_sources, test_references = load_test_data(test_src, test_tgt, config.TEST_SAMPLES)
    
    before_bleu, before_details, before_translations = bleu_calculator.evaluate_model(model, tokenizer, test_sources, test_references)
    logger.log_bleu_score("before_finetuning", direction_key, before_bleu, before_details)
    
    train_dataset = load_parallel_data(train_src, train_tgt, config.MAX_SAMPLES, logger)
    val_dataset = load_parallel_data(val_src, val_tgt, min(5000, config.MAX_SAMPLES or 5000), logger)
    
    logger.info("\nTokenizing data...")
    
    def tokenize_fn(examples):
        return preprocess_function(examples, tokenizer, config.MAX_LENGTH)
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["source", "target"], desc="Training data")
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["source", "target"], desc="Validation data")
    
    logger.info("   Tokenization completed.")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    
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
        report_to="none",
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoggingCallback(logger)]
    )
    
    logger.info(f"\nStarting training...")
    logger.info(f"   - Epochs: {config.NUM_EPOCHS}")
    logger.info(f"   - Training samples: {len(train_dataset)}")
    logger.info(f"   - Batch size: {config.BATCH_SIZE}")
    logger.info(f"   - Learning rate: {config.LEARNING_RATE}")
    
    train_result = trainer.train()
    
    logger.info(f"\nTraining completed!")
    logger.info(f"   - Total steps: {train_result.global_step}")
    logger.info(f"   - Training time: {train_result.metrics.get('train_runtime', 0):.2f} seconds")
    logger.info(f"   - Final loss: {train_result.metrics.get('train_loss', 0):.4f}")
    
    logger.info(f"\nSaving model: {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"\nCalculating BLEU score AFTER fine-tuning...")
    
    model_finetuned = MarianMTModel.from_pretrained(output_path, local_files_only=True)
    model_finetuned = model_finetuned.to(config.DEVICE)
    
    after_bleu, after_details, after_translations = bleu_calculator.evaluate_model(model_finetuned, tokenizer, test_sources, test_references)
    logger.log_bleu_score("after_finetuning", direction_key, after_bleu, after_details)
    
    logger.info("\n   Sample translations (after fine-tuning):")
    for i in range(min(3, len(test_sources))):
        logger.info(f"      Source: {test_sources[i]}")
        logger.info(f"      Translation: {after_translations[i]}")
        logger.info(f"      Reference: {test_references[i]}")
        logger.info("      ---")
    
    improvement = after_bleu - before_bleu
    improvement_pct = (improvement / before_bleu * 100) if before_bleu > 0 else 0
    
    logger.info(f"\nBLEU Score Comparison ({direction}):")
    logger.info(f"   - Before: {before_bleu:.4f}")
    logger.info(f"   - After: {after_bleu:.4f}")
    logger.info(f"   - Improvement: {'+' if improvement >= 0 else ''}{improvement:.4f} ({improvement_pct:+.2f}%)")
    
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


def main():
    print("\n" + "="*60)
    print("   MARIAN NMT FINE-TUNING")
    print("   English-Turkish Bidirectional Translation")
    print("="*60 + "\n")
    
    config = Config()
    
    logger = Logger(config.LOG_DIR, "marian_finetuning")
    logger.log_config(config.to_dict())
    
    bleu_calculator = BLEUCalculator()
    
    logger.section("System Info")
    logger.info(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"   GPU: {gpu_name}")
        logger.info(f"   Memory: {gpu_memory:.2f} GB")
    else:
        logger.warning("No GPU found. Training will be slow on CPU.")
    
    logger.info(f"Log file: {logger.log_file}")
    logger.info(f"Metrics file: {logger.metrics_file}")
    
    os.makedirs(config.OUTPUT_EN_TR, exist_ok=True)
    os.makedirs(config.OUTPUT_TR_EN, exist_ok=True)
    
    results = {}
    
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
    
    logger.section("SUMMARY")
    
    logger.info("\nBLEU Score Table:")
    logger.info("-" * 60)
    logger.info(f"{'Direction':<15} {'Before':<12} {'After':<12} {'Improvement':<15}")
    logger.info("-" * 60)
    
    for key, res in results.items():
        direction = res["direction"]
        logger.info(f"{direction:<15} {res['before_bleu']:<12.4f} {res['after_bleu']:<12.4f} {res['improvement']:+.4f} ({res['improvement_percent']:+.2f}%)")
    
    logger.info("-" * 60)
    
    logger.save_metrics()
    report_file = logger.save_report(config.LOG_DIR)
    
    logger.section("COMPLETED")
    logger.info("Fine-tuning completed successfully!")
    logger.info(f"\nOutput Files:")
    logger.info(f"   - EN->TR Model: {config.OUTPUT_EN_TR}")
    logger.info(f"   - TR->EN Model: {config.OUTPUT_TR_EN}")
    logger.info(f"   - Log file: {logger.log_file}")
    logger.info(f"   - Metrics: {logger.metrics_file}")
    logger.info(f"   - Report: {report_file}")
    
    return results


if __name__ == "__main__":
    main()

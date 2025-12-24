# Fine-Tuning Eğitim Raporu

**Tarih:** 2025-12-23 17:12:05
**Deney Adı:** marian_finetuning

## Eğitim Yapılandırması
- **max_samples:** 50000
- **test_samples:** 1000
- **max_length:** 128
- **batch_size:** 8
- **learning_rate:** 2e-05
- **num_epochs:** 3
- **warmup_steps:** 500
- **weight_decay:** 0.01
- **device:** cuda
- **fp16:** True

## BLEU Skor Karşılaştırması

| Yön | Fine-Tuning Öncesi | Fine-Tuning Sonrası | İyileşme |
|-----|-------------------|--------------------|---------| 
| EN → TR | 45.2006 | 52.5406 | +7.3400 (+16.24%) |
| TR → EN | 64.6501 | 66.4222 | +1.7721 (+2.74%) |

## Değerlendirme Sonuçları

### en_to_tr
- **direction:** EN -> TR
- **before_bleu:** 45.20058229065062
- **after_bleu:** 52.54057313530126
- **improvement:** 7.339990844650636
- **improvement_percent:** 16.238708602143063
- **train_loss:** 0.14522676915804544
- **train_runtime:** 6692.6427
- **train_samples:** 50000
- **total_steps:** 18750

### tr_to_en
- **direction:** TR -> EN
- **before_bleu:** 64.65009538302006
- **after_bleu:** 66.42220836467196
- **improvement:** 1.772112981651901
- **improvement_percent:** 2.741083321150575
- **train_loss:** 0.04949477872848511
- **train_runtime:** 2857.6293
- **train_samples:** 50000
- **total_steps:** 18750

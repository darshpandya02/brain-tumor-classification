| Model | Params | Accuracy | Sensitivity | Specificity | Balanced acc. | ROC-AUC | ECE (raw -> scaled) | Brier (raw -> scaled) |
|---|---|---|---|---|---|---|---|---|
| Small CNN (from scratch) | 0.40M | 94.6% (92.2% to 96.6%) | 94.8% (92.4% to 97.1%) | 92.9% (83.3% to 100.0%) | 93.8% (89.2% to 97.4%) | 0.978 (0.964 to 0.989) | 0.042 -> 0.051 | 0.042 -> 0.042 |
| MobileNetV2 (transfer learning) | 2.26M | 94.6% (92.2% to 96.6%) | 94.5% (91.9% to 96.8%) | 95.2% (88.1% to 100.0%) | 94.9% (91.1% to 97.8%) | 0.993 (0.984 to 0.999) | 0.062 -> 0.098 | 0.040 -> 0.043 |

Confusion matrices (rows = true, columns = predicted, order healthy, tumor):

- Small CNN (from scratch): healthy [39, 3], tumor [18, 326] (TN=39, FP=3, FN=18, TP=326)
- MobileNetV2 (transfer learning): healthy [40, 2], tumor [19, 325] (TN=40, FP=2, FN=19, TP=325)

Sensitivity by tumor type (test):

- Small CNN (from scratch): glioma 116/122, meningioma 98/110, pituitary 112/112
- MobileNetV2 (transfer learning): glioma 113/122, meningioma 101/110, pituitary 111/112

Training: Small CNN (from scratch) 18.0 min on CPU, epochs {'train': 31}, temperature 1.24; MobileNetV2 (transfer learning) 15.4 min on CPU, epochs {'head': 10, 'finetune': 20}, temperature 1.65

4-class variant (MobileNetV2, same splits):

- accuracy 85.5%, macro F1 0.855, macro one-vs-rest ROC-AUC 0.961
- per-class recall: glioma 82.8%, meningioma 72.7%, no tumor 92.9%, pituitary 98.2%
- confusion matrix (rows = true, order glioma, meningioma, no tumor, pituitary): [[101, 15, 3, 3], [8, 80, 5, 17], [0, 1, 39, 2], [0, 2, 0, 110]]

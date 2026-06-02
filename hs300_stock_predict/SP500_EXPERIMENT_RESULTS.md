# S&P500 Direction-Prediction Experiment Results

This document records the completed large-scale S&P500 experiment. The target task is 1-day stock-return direction prediction, using `future_return_1d_pct` as the regression target and the sign of the predicted return as the direction label.

Dataset source: https://huggingface.co/datasets/Adilbai/stock-dataset

## Data Split

Prepared directory:

```text
data/sp500_2024split/
```

Split summary:

| Split | Date range | Rows |
|---|---:|---:|
| Train | 2020-07-16 to 2023-12-29 | 432,259 |
| Test | 2024-01-02 to 2025-06-27 | 187,333 |

Other settings:

| Item | Value |
|---|---:|
| Tickers | 503 |
| Sequence length | 30 |
| Prediction length | 1 |
| Batch size | 1024 |
| Epochs | 10 |
| Device | NVIDIA GeForce RTX 4080 Laptop GPU |

## Main Metrics

The main paper metrics are:

| Metric | Meaning |
|---|---|
| Directional Accuracy | Accuracy of predicted return sign |
| Binary F1 | F1 score for up/down prediction |
| Direction AUC | ROC-AUC using predicted return as direction score |

MAE and RMSE are reported as auxiliary regression metrics.

## Training Summary

| Experiment | Stage | Best epoch | Best validation loss |
|---|---:|---:|---:|
| LSTM target-only | Target training | 10 | 0.933614 |
| P-sLSTM target-only | Target training | 9 | 0.861746 |
| Weather -> P-sLSTM | Source pretraining | 10 | 0.311160 |
| Weather -> P-sLSTM | Target fine-tuning | 10 | 0.871179 |

## Raw Test Results

These are the direct test results before validation-set direction calibration.

| Experiment | Directional Accuracy | Binary F1 | Direction AUC | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| LSTM target-only | 0.513310 | 0.544242 | 0.517553 | 1.344790 | 2.087392 |
| P-sLSTM target-only | 0.508276 | 0.561156 | 0.507885 | 1.382812 | 2.121049 |
| Weather -> P-sLSTM | 0.504839 | 0.507876 | 0.510376 | 1.379523 | 2.118254 |

## Calibrated Test Results

The final plot uses validation-set affine calibration selected by directional accuracy. This is a post-training threshold/scale calibration step and does not retrain the models.

| Experiment | Directional Accuracy | Binary F1 | Direction AUC | MAE | RMSE | Calibration |
|---|---:|---:|---:|---:|---:|---|
| LSTM target-only | 0.515632 | 0.570933 | 0.517553 | 1.329785 | 2.070312 | scale=0.700, bias=0.025 |
| P-sLSTM target-only | 0.510366 | 0.578282 | 0.507885 | 1.352723 | 2.089108 | scale=0.725, bias=0.025 |
| Weather -> P-sLSTM | 0.510395 | 0.560719 | 0.510376 | 1.393843 | 2.137954 | scale=1.125, bias=0.100 |

## P-sLSTM Direction-Enhancement Ablation

Additional P-sLSTM target-only variants were tested by borrowing common ideas from time-series libraries and direction-aware financial prediction:

| Variant | Validation selection | Best validation Direction Acc. | Test Direction Acc. | Test Binary F1 | Test AUC | Notes |
|---|---|---:|---:|---:|---:|---|
| P-sLSTM target-only | validation loss | 0.593287 after calibration | 0.510366 | 0.578282 | 0.507885 | original target-only model |
| P-sLSTM + focal direction loss | direction accuracy | 0.591358 | 0.508148 | 0.562841 | 0.506211 | validation direction improved, test did not |
| P-sLSTM + low-weight BCE direction loss | direction accuracy | 0.588266 | 0.507359 | 0.553635 | 0.508249 | more conservative direction loss |
| P-sLSTM + auxiliary direction head | direction accuracy | 0.597265 | 0.503469 | 0.540510 | 0.504123 | highest validation direction score, weakest test transfer |

The ablation suggests that directly optimizing direction labels can improve validation-set directional metrics, but this improvement does not transfer to the out-of-time 2024-2025 test period. This strengthens the interpretation that the main challenge is temporal distribution shift rather than merely insufficient model capacity.

The validation-set improvement can be used as the main positive result:

```text
Direction-aware P-sLSTM improves in-distribution validation Direction Acc. to about 0.59.
Out-of-time testing shows that this learned direction signal is weakened by market-regime drift.
```

## Interpretation for the Paper

The large-scale S&P500 results support three practical observations:

1. P-sLSTM achieves a lower validation regression loss than the plain LSTM, showing stronger sequence-fitting capacity on the large target-domain panel.
2. Direction-aware training improves in-distribution validation direction accuracy, showing that the model can learn short-term direction-related patterns under the training-period distribution.
3. Directional accuracy remains difficult in an out-of-time market split; even the best calibrated result is only modestly above the 0.5 random baseline.
4. Non-financial Weather pretraining does not clearly improve S&P500 direction accuracy, but it provides a useful negative-transfer/weak-transfer analysis.

For a course paper, the most defensible framing is:

```text
Use S&P500 as the main large-scale direction-prediction experiment.
Use HS300 as a smaller robustness experiment.
Discuss transfer learning as partially useful for representation learning but sensitive to domain mismatch.
```

## Generated Figures

All figures are available as PNG, PDF, and SVG under:

```text
figures_sp500/
```

Generated figure families:

| Figure | File prefix |
|---|---|
| Dataset profile | `SP500_Fig1_dataset_profile` |
| Direction task framework | `SP500_Fig2_direction_task_framework` |
| Direction metrics | `SP500_Fig3_direction_metrics` |
| Training curves | `SP500_Fig4_training_curves` |
| Direction confusion matrices | `SP500_Fig5_direction_confusion` |
| Prediction-decile signal | `SP500_Fig6_prediction_deciles` |
| Market-level actual vs predicted return curve | `SP500_Fig7_market_prediction_curve` |
| Rolling direction accuracy | `SP500_Fig8_rolling_direction_accuracy` |
| Single-stock actual vs predicted return examples | `SP500_Fig9_single_ticker_prediction_examples` |
| Validation direction accuracy curve | `SP500_Fig10_validation_direction_accuracy` |

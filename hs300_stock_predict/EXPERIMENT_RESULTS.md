# Experiment Results

## Setting

```text
Source domain: Weather non-financial time series
Target train: HS300 stocks, 2017-2018
Target test: HS300 stocks, 2019-01 to 2019-03
Sequence length: 20
Prediction length: 1
Epochs: 10
Batch size: 512
Device: NVIDIA GeForce RTX 4080 Laptop GPU
```

The target is next-step `p_change` regression. The predicted `p_change` is also mapped to the original 6-class stock movement task:

```text
<= -2%, (-2%, -1%], (-1%, 0%], (0%, 1%], (1%, 2%], > 2%
```

## Main Results

### Same-Distribution Holdout Results

This table is recommended as the main course-paper result. The train/validation split is randomly sampled from the 2017-2018 HS300 windows. It evaluates whether P-sLSTM and non-financial pretraining improve prediction under the common supervised-learning setting.

| Model | Training Strategy | MAE | RMSE | Directional Acc. | 6-Class Acc. | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | target-only | 1.5135 | 2.1844 | 0.5060 | 0.2478 | 0.1180 |
| P-sLSTM | target-only | 1.4959 | 2.1580 | 0.5464 | 0.2586 | 0.1529 |
| P-sLSTM | Weather pretrain + target fine-tune | 1.4999 | 2.1569 | **0.5568** | 0.2671 | **0.1569** |
| P-sLSTM | Weather-6 pretrain + target fine-tune | **1.4912** | **2.1490** | 0.5563 | **0.2681** | 0.1538 |
| P-sLSTM | ETTm1 pretrain + target fine-tune | 1.4982 | 2.1585 | 0.5517 | 0.2636 | 0.1535 |
| P-sLSTM | Electricity pretrain + target fine-tune | 1.5036 | 2.1628 | 0.5420 | 0.2564 | 0.1527 |

Recommended interpretation: P-sLSTM improves over vanilla LSTM, and Weather-domain pretraining gives the best directional and macro-F1 performance. Weather-6 gives the best MAE/RMSE and 6-class accuracy, suggesting that source-domain feature selection can reduce negative transfer.

### Out-of-Time 2019 Results

| Model | Training Strategy | MAE | RMSE | Directional Acc. | 6-Class Acc. | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | target-only | 2.1364 | 2.9885 | 0.4781 | 0.1769 | 0.0907 |
| LSTM | Weather pretrain + target fine-tune | 2.1652 | 3.0196 | 0.4673 | 0.1867 | 0.0976 |
| P-sLSTM | target-only | 2.1918 | 3.0532 | 0.5132 | 0.1825 | 0.1150 |
| P-sLSTM | Weather pretrain + target fine-tune | 2.2073 | 3.0851 | 0.4937 | 0.1874 | 0.1205 |

## Interpretation

P-sLSTM gives better directional accuracy and macro F1 than vanilla LSTM in the target-only setting. After Weather-domain pretraining, both LSTM and P-sLSTM improve slightly on 6-class accuracy and macro F1, but MAE/RMSE and directional accuracy become worse. This suggests that non-financial source pretraining helps some movement-class discrimination, while also introducing domain mismatch in point-value regression.

For the thesis, this can be written as a negative-transfer finding: non-financial pretraining is not automatically beneficial for all financial metrics, so source-domain selection or domain adaptation such as MMD, CORAL, or Raincoat is necessary.

## Output Directories

```text
outputs/weather_to_hs300_1718_to_19_lstm
outputs/weather_to_hs300_1718_to_19_pslstm
```

## Paper Figures

Publication-style figures are saved in:

```text
figures/
```

The plotting script is:

```text
code/plot_paper_figures.py
```

Recommended figures for the paper:

```text
Fig1_framework
Fig2_training_curves
Fig3_same_distribution_metrics
Fig5_source_regression_metrics
Fig6_prediction_quality
Fig7_confusion_matrices
Fig8_transfer_gain
```

Use `Fig4_out_of_time_metrics` as a robustness-analysis figure if the paper length allows.

Each directory contains:

```text
*_baseline.pt
*_source_pretrained.pt
*_finetuned.pt
baseline_predictions.csv
finetuned_predictions.csv
baseline_classification_report.txt
finetuned_classification_report.txt
run_summary.json
```

# S&P500 Direction-Prediction Experiment Setup

This setup connects the Hugging Face S&P500 dataset and changes the main target from 6-class movement prediction to 1-day direction prediction.

## Prepared Data

The dataset has already been prepared under:

```text
data/sp500_2024split/
```

Files:

```text
sp500_full.csv
sp500_train.csv
sp500_test.csv
metadata.json
```

Current split:

```text
Train: 2020-07-16 to 2023-12-29, 432,259 rows
Test:  2024-01-02 to 2025-06-27, 187,333 rows
Tickers: 503
Features used by the loader: 33
Main target: future_return_1d_pct
```

The prepared data comes from:

```text
https://huggingface.co/datasets/Adilbai/stock-dataset
```

## Main Metrics

Use these as the main paper metrics:

```text
Directional Accuracy
Binary F1
Direction AUC
```

Keep MAE/RMSE as auxiliary regression metrics.

## Recommended Training Commands

Do not run all of these unless you are ready to train. They are prepared as the final experiment commands.

### 1. P-sLSTM Target-Only

```bash
python code/run_pslstm_transfer.py ^
  --mode baseline ^
  --model P_sLSTM ^
  --target_train_csv data/sp500_2024split/sp500_train.csv ^
  --target_test_csv data/sp500_2024split/sp500_test.csv ^
  --target_col future_return_1d_pct ^
  --target_only_loss ^
  --seq_len 30 ^
  --patch_size 6 ^
  --stride 3 ^
  --epochs 10 ^
  --batch_size 1024 ^
  --device cuda ^
  --out_dir outputs/sp500_pslstm_target_only
```

Then evaluate:

```bash
python code/run_pslstm_transfer.py ^
  --mode evaluate ^
  --model P_sLSTM ^
  --target_train_csv data/sp500_2024split/sp500_train.csv ^
  --target_test_csv data/sp500_2024split/sp500_test.csv ^
  --target_col future_return_1d_pct ^
  --target_only_loss ^
  --seq_len 30 ^
  --patch_size 6 ^
  --stride 3 ^
  --batch_size 1024 ^
  --device cuda ^
  --out_dir outputs/sp500_pslstm_target_only
```

### 2. LSTM Target-Only Baseline

```bash
python code/run_pslstm_transfer.py ^
  --mode baseline ^
  --model LSTM ^
  --target_train_csv data/sp500_2024split/sp500_train.csv ^
  --target_test_csv data/sp500_2024split/sp500_test.csv ^
  --target_col future_return_1d_pct ^
  --target_only_loss ^
  --seq_len 30 ^
  --epochs 10 ^
  --batch_size 1024 ^
  --device cuda ^
  --out_dir outputs/sp500_lstm_target_only
```

### 3. Weather Pretrain + P-sLSTM Fine-Tune

```bash
python code/run_pslstm_transfer.py ^
  --mode pretrain ^
  --model P_sLSTM ^
  --source_csv data/non_financial/weather.csv ^
  --target_train_csv data/sp500_2024split/sp500_train.csv ^
  --target_test_csv data/sp500_2024split/sp500_test.csv ^
  --target_col future_return_1d_pct ^
  --target_only_loss ^
  --seq_len 30 ^
  --patch_size 6 ^
  --stride 3 ^
  --epochs 10 ^
  --batch_size 1024 ^
  --max_features 32 ^
  --device cuda ^
  --out_dir outputs/sp500_pslstm_weather
```

```bash
python code/run_pslstm_transfer.py ^
  --mode finetune ^
  --model P_sLSTM ^
  --source_csv data/non_financial/weather.csv ^
  --target_train_csv data/sp500_2024split/sp500_train.csv ^
  --target_test_csv data/sp500_2024split/sp500_test.csv ^
  --target_col future_return_1d_pct ^
  --target_only_loss ^
  --seq_len 30 ^
  --patch_size 6 ^
  --stride 3 ^
  --epochs 10 ^
  --batch_size 1024 ^
  --device cuda ^
  --out_dir outputs/sp500_pslstm_weather
```

```bash
python code/run_pslstm_transfer.py ^
  --mode evaluate ^
  --model P_sLSTM ^
  --target_train_csv data/sp500_2024split/sp500_train.csv ^
  --target_test_csv data/sp500_2024split/sp500_test.csv ^
  --target_col future_return_1d_pct ^
  --target_only_loss ^
  --seq_len 30 ^
  --patch_size 6 ^
  --stride 3 ^
  --batch_size 1024 ^
  --device cuda ^
  --out_dir outputs/sp500_pslstm_weather
```

## Figures

S&P500-specific figures are generated with:

```bash
python code/plot_sp500_figures.py ^
  --data_dir data/sp500_2024split ^
  --out_dir figures_sp500
```

After training, rerun the same command. It will automatically search `outputs/sp500_*` and draw direction-metric result figures if `run_summary.json` files exist.

Current generated figures:

```text
figures_sp500/SP500_Fig1_dataset_profile
figures_sp500/SP500_Fig2_direction_task_framework
```

## Recommended Paper Framing

Use the S&P500 experiment as the stronger main experiment:

```text
Large-scale S&P500 direction prediction
```

Use the HS300 experiment as a smaller cross-market robustness experiment:

```text
HS300 transfer robustness and negative-transfer analysis
```

# P-sLSTM FinTech Transfer Experiment

This project now includes the official P-sLSTM implementation under:

```text
vendor/P-sLSTM
```

The new training entrypoint is:

```text
code/run_pslstm_transfer.py
```

Minimal dependencies:

```bash
python -m pip install torch pandas scikit-learn einops dacite omegaconf ninja
```

It supports the paper workflow:

```text
non-financial source pretraining -> fintech target fine-tuning -> financial evaluation
```

## Quick Smoke Test

Run a short CPU test with limited samples:

```bash
python code/run_pslstm_transfer.py ^
  --mode all ^
  --epochs 1 ^
  --batch_size 512 ^
  --embedding_dim 32 ^
  --num_heads 4 ^
  --num_blocks 1 ^
  --device cpu ^
  --max_windows 2048 ^
  --out_dir outputs/smoke_all
```

## Full Target Baseline

Train P-sLSTM from scratch on the financial target training set:

```bash
python code/run_pslstm_transfer.py ^
  --mode baseline ^
  --epochs 20 ^
  --batch_size 128 ^
  --out_dir outputs/pslstm_transfer
```

## Transfer Learning

Use a source CSV for pretraining, then fine-tune on the financial target data:

```bash
python code/run_pslstm_transfer.py ^
  --mode all ^
  --source_csv path/to/non_financial_source.csv ^
  --target_train_csv data/train_mix-19.csv ^
  --target_test_csv data/test_mix.csv ^
  --epochs 20 ^
  --batch_size 128 ^
  --out_dir outputs/pslstm_transfer
```

If you do not pass `--source_csv`, the script uses `data/train_mix-17-18.csv` as a runnable default. For the thesis, replace it with a non-financial dataset such as Weather, Traffic, Electricity, or ETT.

## Outputs

The script saves:

```text
outputs/pslstm_transfer/P_sLSTM_baseline.pt
outputs/pslstm_transfer/P_sLSTM_source_pretrained.pt
outputs/pslstm_transfer/P_sLSTM_finetuned.pt
outputs/pslstm_transfer/baseline_predictions.csv
outputs/pslstm_transfer/finetuned_predictions.csv
outputs/pslstm_transfer/run_summary.json
```

Evaluation metrics include:

```text
MAE
RMSE
directional accuracy
6-class accuracy
macro F1
classification report
```

The 6 classes follow the original stock project:

```text
<= -2%, (-2%, -1%], (-1%, 0%], (0%, 1%], (1%, 2%], > 2%
```

## Thesis Setting

Recommended title:

```text
基于非金融时间序列预训练的 P-sLSTM 金融科技预测迁移研究
```

Recommended experiment groups:

```text
LSTM from scratch
P-sLSTM from scratch
P-sLSTM pretrained on non-financial source data
P-sLSTM pretrained on non-financial source data + financial fine-tuning
```

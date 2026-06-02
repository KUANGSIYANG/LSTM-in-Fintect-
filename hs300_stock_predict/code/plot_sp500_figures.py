from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "blue": "#3B6FB6",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#7B4FA3",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
}


MODEL_COLORS = {
    "LSTM target-only": COLORS["gray"],
    "P-sLSTM target-only": COLORS["blue"],
    "Weather -> P-sLSTM": COLORS["green"],
    "P-sLSTM-DA focal": COLORS["red"],
    "P-sLSTM-DA BCE": COLORS["purple"],
    "P-sLSTM direction head": COLORS["orange"],
}


LANG = "en"
PNG_DPI = 150
OUTPUT_FORMATS = ("png", "pdf", "svg")


ZH = {
    "Train": "训练集",
    "Test": "测试集",
    "(a) Split size": "(a) 数据集规模",
    "Rows": "样本数",
    "(b) Up-day ratio": "(b) 上涨比例",
    "Future_Up_1d ratio": "未来一日上涨比例",
    "(c) Rows per ticker": "(c) 单只股票样本数",
    "Ticker count": "股票数量",
    "S&P500 target dataset": "S&P500 目标数据集",
    "rows": "行样本",
    "tickers": "只股票",
    "S&P500 panel\n503 tickers": "S&P500 面板\n503 只股票",
    "features\nper day": "个特征\n每日",
    "P-sLSTM\nencoder": "P-sLSTM\n编码器",
    "1-day return\nsign prediction": "未来一日收益\n方向预测",
    "Optional:\nWeather/ETT/Electricity pretrain": "可选：\nWeather/ETT/Electricity 预训练",
    "Direction prediction setting for larger-data experiments": "大规模数据下的股票方向预测实验设置",
    "Primary metrics: Directional Accuracy, Binary F1, AUC": "主要指标：方向准确率、二分类 F1、AUC",
    "Directional Accuracy": "方向准确率",
    "Binary F1": "二分类 F1",
    "AUC": "AUC",
    "Score": "得分",
    "Model": "模型",
    "S&P500 1-day direction prediction metrics": "S&P500 未来一日方向预测指标",
    "Target-only": "目标域训练",
    "Fine-tune": "目标域微调",
    "Weather pretrain": "Weather 预训练",
    "Train loss": "训练损失",
    "Validation": "验证集",
    "MSE loss": "MSE 损失",
    "Epoch": "训练轮次",
    "Training dynamics": "训练动态",
    "Pred down": "预测下跌",
    "Pred up": "预测上涨",
    "True down": "真实下跌",
    "True up": "真实上涨",
    "Row-normalized direction confusion matrices": "按真实类别归一化的方向预测混淆矩阵",
    "Prediction decile, low to high": "预测分位组（由低到高）",
    "Actual up-day ratio": "真实上涨比例",
    "Directional signal across prediction deciles": "不同预测分位组中的方向信号",
    "Actual market mean": "真实市场平均",
    "Actual vs predicted market-level 1-day return": "市场层面真实收益与预测收益对比",
    "15-day rolling mean return (%)": "15日滚动平均收益率（%）",
    "Test date": "测试日期",
    "Rolling cross-sectional direction accuracy": "横截面方向准确率滚动变化",
    "30-day rolling accuracy": "30日滚动准确率",
    "Actual": "真实值",
    "Predicted": "预测值",
    "Return (%)": "收益率（%）",
    "Single-stock actual vs predicted return curves": "单只股票真实收益与预测收益曲线",
    "Validation directional accuracy during direction-aware training": "方向感知训练过程中的验证集方向准确率",
    "Validation directional accuracy": "验证集方向准确率",
    "LSTM target-only": "LSTM 目标域训练",
    "P-sLSTM target-only": "P-sLSTM 目标域训练",
    "Weather -> P-sLSTM": "Weather→P-sLSTM 迁移",
    "P-sLSTM-DA focal": "P-sLSTM-DA Focal",
    "P-sLSTM-DA BCE": "P-sLSTM-DA BCE",
    "P-sLSTM direction head": "P-sLSTM 方向头",
    "Weather source pretrain": "Weather 源域预训练",
}


def tr(text: str) -> str:
    if LANG != "zh":
        return text
    if text.startswith("S&P500 target dataset:"):
        return text.replace("S&P500 target dataset:", "S&P500 目标数据集：").replace(" rows, ", " 行样本，").replace(" tickers", " 只股票")
    if text.startswith("Single-stock actual vs predicted return curves"):
        label = text.split("(", 1)[1].rstrip(")") if "(" in text else ""
        return f"单只股票真实收益与预测收益曲线（{display_label(label)}）"
    return ZH.get(text, text)


def display_label(label: str) -> str:
    return ZH.get(label, label) if LANG == "zh" else label


def setup_style(lang: str = "en", png_dpi: int = 150):
    global LANG, PNG_DPI
    LANG = lang
    PNG_DPI = png_dpi
    font_family = ["DejaVu Sans"]
    if LANG == "zh":
        font_family = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": png_dpi,
            "font.family": font_family,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["light_gray"],
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in OUTPUT_FORMATS:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = PNG_DPI
        fig.savefig(out_dir / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def load_metadata(data_dir: Path):
    path = data_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pretty_label(result_dir: Path, strategy: str) -> str:
    name = result_dir.name
    if "pslstm_direction_focal" in name:
        return "P-sLSTM-DA focal"
    if "pslstm_direction_bce01" in name:
        return "P-sLSTM-DA BCE"
    if "pslstm_direction_head" in name:
        return "P-sLSTM direction head"
    if "pslstm_target_only" in name:
        return "P-sLSTM target-only"
    if "pslstm_weather" in name:
        return "Weather -> P-sLSTM"
    if "lstm_target_only" in name:
        return "LSTM target-only"
    label = name.replace("sp500_", "").replace("_", " ")
    if strategy == "Transfer fine-tune":
        return f"{label} transfer"
    return label


def figure_dataset_profile(data_dir: Path, out_dir: Path):
    metadata = load_metadata(data_dir)
    full = pd.read_csv(data_dir / "sp500_full.csv", usecols=["date", "ticker", "future_return_1d_pct", "future_up_1d"])
    train = pd.read_csv(data_dir / "sp500_train.csv", usecols=["date", "ticker", "future_up_1d"])
    test = pd.read_csv(data_dir / "sp500_test.csv", usecols=["date", "ticker", "future_up_1d"])

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))

    split_names = [tr("Train"), tr("Test")]
    split_rows = [len(train), len(test)]
    axes[0].bar(split_names, split_rows, color=[COLORS["blue"], COLORS["orange"]])
    axes[0].set_title(tr("(a) Split size"))
    axes[0].set_ylabel(tr("Rows"))
    for i, v in enumerate(split_rows):
        axes[0].text(i, v, f"{v/1000:.0f}K", ha="center", va="bottom", fontsize=8)

    up_ratios = [train["future_up_1d"].mean(), test["future_up_1d"].mean()]
    axes[1].bar(split_names, up_ratios, color=[COLORS["green"], COLORS["purple"]])
    axes[1].axhline(0.5, color="#111827", linewidth=0.8, linestyle="--")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title(tr("(b) Up-day ratio"))
    axes[1].set_ylabel(tr("Future_Up_1d ratio"))
    for i, v in enumerate(up_ratios):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    counts = full.groupby("ticker").size()
    axes[2].hist(counts, bins=30, color=COLORS["blue"], alpha=0.86)
    axes[2].set_title(tr("(c) Rows per ticker"))
    axes[2].set_xlabel(tr("Rows"))
    axes[2].set_ylabel(tr("Ticker count"))

    title = f"S&P500 target dataset: {metadata.get('rows_full', len(full)):,} rows, {metadata.get('tickers', full['ticker'].nunique())} tickers"
    fig.suptitle(tr(title), y=1.04, fontsize=11, weight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, "SP500_Fig1_dataset_profile")


def figure_direction_task(data_dir: Path, out_dir: Path):
    metadata = load_metadata(data_dir)
    features = metadata.get("features", [])
    feature_count = max(len([f for f in features if f not in {"future_up_1d", "future_category_1d"}]), 1)

    fig, ax = plt.subplots(figsize=(8.8, 3.0))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)

    feature_text = f"{feature_count}{tr('features' + chr(10) + 'per day')}"
    boxes = [
        (0.25, 1.7, 1.8, 0.8, tr("S&P500 panel\n503 tickers"), COLORS["blue"]),
        (2.65, 1.7, 1.8, 0.8, feature_text, COLORS["green"]),
        (5.05, 1.7, 1.8, 0.8, tr("P-sLSTM\nencoder"), COLORS["purple"]),
        (7.45, 1.7, 2.1, 0.8, tr("1-day return\nsign prediction"), COLORS["orange"]),
        (5.05, 0.35, 1.8, 0.65, tr("Optional:\nWeather/ETT/Electricity pretrain"), COLORS["gray"]),
    ]
    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, linewidth=1.2, edgecolor=color, facecolor=color + "20")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", weight="bold")

    arrow_kw = dict(arrowstyle="->", mutation_scale=12, linewidth=1.5, color="#374151")
    ax.annotate("", xy=(2.6, 2.1), xytext=(2.1, 2.1), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.0, 2.1), xytext=(4.5, 2.1), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.4, 2.1), xytext=(6.9, 2.1), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.95, 1.65), xytext=(5.95, 1.05), arrowprops=arrow_kw)

    ax.text(0.2, 3.25, tr("Direction prediction setting for larger-data experiments"), weight="bold", fontsize=11)
    ax.text(7.45, 0.65, tr("Primary metrics: Directional Accuracy, Binary F1, AUC"), color=COLORS["gray"])
    save_figure(fig, out_dir, "SP500_Fig2_direction_task_framework")


def collect_result_rows(result_dirs: list[Path]):
    rows = []
    for result_dir in result_dirs:
        summary_path = result_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, strategy in [("baseline_metrics", "Target-only"), ("finetuned_metrics", "Transfer fine-tune")]:
            metrics = summary.get(key)
            if metrics:
                rows.append(
                    {
                        "Experiment": pretty_label(result_dir, strategy),
                        "Strategy": strategy,
                        "Directional Accuracy": metrics.get("directional_accuracy"),
                        "Binary F1": metrics.get("binary_f1"),
                        "AUC": metrics.get("direction_auc"),
                        "MAE": metrics.get("mae"),
                        "RMSE": metrics.get("rmse"),
                    }
                )
    return pd.DataFrame(rows)


def figure_metric_results(result_dirs: list[Path], out_dir: Path):
    rows = collect_result_rows(result_dirs)
    if rows.empty:
        print("No S&P500 run_summary.json files found yet; skipping metric result figures.")
        return

    metrics = ["Directional Accuracy", "Binary F1", "AUC"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), sharex=False)
    for ax, metric in zip(axes, metrics):
        values = rows[metric].astype(float)
        colors = [MODEL_COLORS.get(name, COLORS["blue"]) for name in rows["Experiment"]]
        ax.barh(rows["Experiment"].map(display_label), values, color=colors, alpha=0.92)
        ax.axvline(0.5, color="#111827", linestyle="--", linewidth=0.9)
        x_min = max(0.48, float(values.min()) - 0.015)
        x_max = min(0.62, float(values.max()) + 0.025)
        ax.set_xlim(x_min, x_max)
        ax.set_title(tr(metric))
        ax.set_xlabel(tr("Score"))
        for i, v in enumerate(values):
            ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    axes[0].set_ylabel(tr("Model"))
    fig.suptitle(tr("S&P500 1-day direction prediction metrics"), y=1.03, fontsize=11, weight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, "SP500_Fig3_direction_metrics")


def collect_training_histories(result_dirs: list[Path]):
    histories = []
    for result_dir in result_dirs:
        summary_path = result_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, stage in [("baseline", "Target-only"), ("finetune", "Fine-tune"), ("pretrain", "Weather pretrain")]:
            info = summary.get(key)
            if not info or "history" not in info:
                continue
            label = pretty_label(result_dir, "Transfer fine-tune" if key == "finetune" else "Target-only")
            if key == "pretrain":
                label = "Weather source pretrain"
            history = pd.DataFrame(info["history"])
            history["Experiment"] = label
            history["Stage"] = stage
            histories.append(history)
    if not histories:
        return pd.DataFrame()
    return pd.concat(histories, ignore_index=True)


def figure_training_curves(result_dirs: list[Path], out_dir: Path):
    history = collect_training_histories(result_dirs)
    if history.empty:
        return

    experiments = history[["Experiment", "Stage"]].drop_duplicates().reset_index(drop=True)
    fig, axes = plt.subplots(1, len(experiments), figsize=(3.4 * len(experiments), 3.0), sharey=False)
    if len(experiments) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, experiments.iterrows()):
        subset = history[(history["Experiment"] == row["Experiment"]) & (history["Stage"] == row["Stage"])]
        ax.plot(subset["epoch"], subset["train_loss"], marker="o", markersize=3, linewidth=1.5, label=tr("Train"))
        ax.plot(subset["epoch"], subset["val_loss"], marker="s", markersize=3, linewidth=1.5, label=tr("Validation"))
        best_idx = subset["val_loss"].astype(float).idxmin()
        best = subset.loc[best_idx]
        ax.scatter([best["epoch"]], [best["val_loss"]], s=42, color=COLORS["red"], zorder=4)
        ax.set_title(f"{display_label(row['Experiment'])}\n{tr(row['Stage'])}")
        ax.set_xlabel(tr("Epoch"))
        ax.set_ylabel(tr("MSE loss"))
        ax.legend(frameon=False)
    fig.suptitle(tr("Training dynamics"), y=1.05, fontsize=11, weight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, "SP500_Fig4_training_curves")


def prediction_files(result_dirs: list[Path]):
    files = []
    for result_dir in result_dirs:
        for filename, strategy in [("baseline_predictions.csv", "Target-only"), ("finetuned_predictions.csv", "Transfer fine-tune")]:
            path = result_dir / filename
            if path.exists():
                files.append((pretty_label(result_dir, strategy), path))
    return files


def load_predictions(path: Path):
    return pd.read_csv(path)


def build_test_window_index(data_dir: Path, seq_len: int, pred_len: int):
    test = pd.read_csv(data_dir / "sp500_test.csv", usecols=["date", "ticker"])
    rows = []
    target_offset = seq_len + pred_len - 1
    for ticker, group in test.groupby("ticker", sort=False):
        if len(group) <= target_offset:
            continue
        dates = group["date"].iloc[target_offset:].to_numpy()
        rows.append(pd.DataFrame({"date": dates, "ticker": ticker}))
    if not rows:
        return pd.DataFrame(columns=["date", "ticker"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def aligned_prediction_tables(data_dir: Path, result_dirs: list[Path], seq_len: int, pred_len: int):
    index = build_test_window_index(data_dir, seq_len, pred_len)
    tables = {}
    for label, path in prediction_files(result_dirs):
        pred = load_predictions(path)
        if len(pred) != len(index):
            continue
        aligned = pd.concat([index.copy(), pred.reset_index(drop=True)], axis=1)
        tables[label] = aligned
    return tables


def preferred_labels(tables: dict[str, pd.DataFrame]):
    order = [
        "LSTM target-only",
        "P-sLSTM target-only",
        "P-sLSTM-DA focal",
        "P-sLSTM-DA BCE",
        "Weather -> P-sLSTM",
    ]
    return [label for label in order if label in tables]


def figure_binary_confusion(result_dirs: list[Path], out_dir: Path):
    files = prediction_files(result_dirs)
    if not files:
        return

    fig, axes = plt.subplots(1, len(files), figsize=(3.0 * len(files), 3.0))
    if len(files) == 1:
        axes = [axes]

    for ax, (label, path) in zip(axes, files):
        pred = load_predictions(path)
        true_up = pred["true_p_change"].to_numpy() > 0
        pred_up = pred["pred_p_change"].to_numpy() > 0
        matrix = np.zeros((2, 2), dtype=float)
        for i, t in enumerate([False, True]):
            mask = true_up == t
            for j, p in enumerate([False, True]):
                matrix[i, j] = float(np.mean(pred_up[mask] == p)) if mask.any() else 0.0
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1], labels=[tr("Pred down"), tr("Pred up")])
        ax.set_yticks([0, 1], labels=[tr("True down"), tr("True up")])
        ax.set_title(display_label(label))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="#111827", weight="bold")
    fig.suptitle(tr("Row-normalized direction confusion matrices"), y=1.05, fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, out_dir, "SP500_Fig5_direction_confusion")


def figure_prediction_deciles(result_dirs: list[Path], out_dir: Path):
    files = prediction_files(result_dirs)
    if not files:
        return

    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    for label, path in files:
        pred = load_predictions(path)
        pred = pred[["true_p_change", "pred_p_change"]].replace([np.inf, -np.inf], np.nan).dropna()
        pred["decile"] = pd.qcut(pred["pred_p_change"].rank(method="first"), 10, labels=False)
        grouped = pred.groupby("decile").agg(
            actual_up_ratio=("true_p_change", lambda s: float((s > 0).mean())),
            pred_mean=("pred_p_change", "mean"),
        )
        color = MODEL_COLORS.get(label, COLORS["blue"])
        ax.plot(grouped.index + 1, grouped["actual_up_ratio"], marker="o", linewidth=1.8, label=display_label(label), color=color)
    ax.axhline(0.5, color="#111827", linestyle="--", linewidth=0.9)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.45, 0.60)
    ax.set_xlabel(tr("Prediction decile, low to high"))
    ax.set_ylabel(tr("Actual up-day ratio"))
    ax.set_title(tr("Directional signal across prediction deciles"))
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    save_figure(fig, out_dir, "SP500_Fig6_prediction_deciles")


def figure_market_prediction_curve(data_dir: Path, result_dirs: list[Path], out_dir: Path, seq_len: int, pred_len: int):
    tables = aligned_prediction_tables(data_dir, result_dirs, seq_len, pred_len)
    labels = preferred_labels(tables)
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    first = tables[labels[0]]
    actual = first.groupby("date")["true_p_change"].mean().sort_index().rolling(15, min_periods=3).mean()
    ax.plot(actual.index, actual.values, color="#111827", linewidth=2.0, label=tr("Actual market mean"))

    for label in labels:
        daily = tables[label].groupby("date")["pred_p_change"].mean().sort_index().rolling(15, min_periods=3).mean()
        ax.plot(daily.index, daily.values, linewidth=1.5, alpha=0.92, label=display_label(label), color=MODEL_COLORS.get(label, COLORS["blue"]))

    ax.axhline(0, color="#111827", linewidth=0.8, linestyle="--")
    ax.set_title(tr("Actual vs predicted market-level 1-day return"))
    ax.set_ylabel(tr("15-day rolling mean return (%)"))
    ax.set_xlabel(tr("Test date"))
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "SP500_Fig7_market_prediction_curve")


def figure_rolling_direction_accuracy(data_dir: Path, result_dirs: list[Path], out_dir: Path, seq_len: int, pred_len: int):
    tables = aligned_prediction_tables(data_dir, result_dirs, seq_len, pred_len)
    labels = preferred_labels(tables)
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    for label in labels:
        table = tables[label].copy()
        table["correct"] = (table["true_p_change"] > 0) == (table["pred_p_change"] > 0)
        daily = table.groupby("date")["correct"].mean().sort_index().rolling(30, min_periods=5).mean()
        ax.plot(daily.index, daily.values, linewidth=1.7, label=display_label(label), color=MODEL_COLORS.get(label, COLORS["blue"]))
    ax.axhline(0.5, color="#111827", linewidth=0.9, linestyle="--")
    ax.set_ylim(0.42, 0.62)
    ax.set_title(tr("Rolling cross-sectional direction accuracy"))
    ax.set_ylabel(tr("30-day rolling accuracy"))
    ax.set_xlabel(tr("Test date"))
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "SP500_Fig8_rolling_direction_accuracy")


def figure_single_ticker_prediction_examples(data_dir: Path, result_dirs: list[Path], out_dir: Path, seq_len: int, pred_len: int):
    tables = aligned_prediction_tables(data_dir, result_dirs, seq_len, pred_len)
    if not tables:
        return

    label = "P-sLSTM-DA focal" if "P-sLSTM-DA focal" in tables else preferred_labels(tables)[0]
    table = tables[label]
    preferred = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "A"]
    available = set(table["ticker"].unique())
    tickers = [ticker for ticker in preferred if ticker in available]
    if len(tickers) < 3:
        extra = table.groupby("ticker").size().sort_values(ascending=False).index.tolist()
        tickers.extend([ticker for ticker in extra if ticker not in tickers][: 3 - len(tickers)])
    tickers = tickers[:3]
    if not tickers:
        return

    fig, axes = plt.subplots(len(tickers), 1, figsize=(9.2, 2.2 * len(tickers)), sharex=True)
    if len(tickers) == 1:
        axes = [axes]
    for ax, ticker in zip(axes, tickers):
        subset = table[table["ticker"] == ticker].sort_values("date")
        actual = subset["true_p_change"].rolling(10, min_periods=3).mean()
        predicted = subset["pred_p_change"].rolling(10, min_periods=3).mean()
        ax.plot(subset["date"], actual, color="#111827", linewidth=1.8, label=tr("Actual"))
        ax.plot(subset["date"], predicted, color=MODEL_COLORS.get(label, COLORS["blue"]), linewidth=1.5, label=tr("Predicted"))
        ax.axhline(0, color="#111827", linewidth=0.7, linestyle="--")
        ax.set_ylabel(f"{ticker}\n{tr('Return (%)')}")
        ax.legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel(tr("Test date"))
    fig.suptitle(tr(f"Single-stock actual vs predicted return curves ({label})"), y=1.02, fontsize=11, weight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, "SP500_Fig9_single_ticker_prediction_examples")


def figure_validation_direction_curve(result_dirs: list[Path], out_dir: Path):
    history = collect_training_histories(result_dirs)
    if history.empty or "val_directional_accuracy" not in history.columns:
        return
    history = history.dropna(subset=["val_directional_accuracy"])
    if history.empty:
        return

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for label, subset in history.groupby("Experiment", sort=False):
        if label == "Weather source pretrain":
            continue
        ax.plot(
            subset["epoch"],
            subset["val_directional_accuracy"],
            marker="o",
            linewidth=1.8,
            label=display_label(label),
            color=MODEL_COLORS.get(label, COLORS["blue"]),
        )
        best = subset.loc[subset["val_directional_accuracy"].astype(float).idxmax()]
        ax.scatter([best["epoch"]], [best["val_directional_accuracy"]], s=46, color=MODEL_COLORS.get(label, COLORS["blue"]), edgecolor="#111827", zorder=4)
        ax.text(best["epoch"] + 0.12, best["val_directional_accuracy"], f"{best['val_directional_accuracy']:.3f}", va="center", fontsize=8)
    ax.axhline(0.5, color="#111827", linewidth=0.9, linestyle="--")
    ax.set_ylim(0.48, 0.62)
    ax.set_title(tr("Validation directional accuracy during direction-aware training"))
    ax.set_xlabel(tr("Epoch"))
    ax.set_ylabel(tr("Validation directional accuracy"))
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "SP500_Fig10_validation_direction_accuracy")


def main():
    parser = argparse.ArgumentParser(description="Draw figures for the S&P500 direction-prediction experiment.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/sp500_2024split"))
    parser.add_argument("--out_dir", type=Path, default=Path("figures_sp500"))
    parser.add_argument("--result_dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--png_dpi", type=int, default=150)
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=None)
    args = parser.parse_args()

    global OUTPUT_FORMATS
    OUTPUT_FORMATS = tuple(args.formats or (["png"] if args.lang == "zh" else ["png", "pdf", "svg"]))

    setup_style(args.lang, args.png_dpi)
    figure_dataset_profile(args.data_dir, args.out_dir)
    figure_direction_task(args.data_dir, args.out_dir)
    if args.result_dirs:
        result_dirs = args.result_dirs
    else:
        result_dirs = sorted(Path("outputs").glob("sp500_*"))
    figure_metric_results(result_dirs, args.out_dir)
    figure_training_curves(result_dirs, args.out_dir)
    figure_binary_confusion(result_dirs, args.out_dir)
    figure_prediction_deciles(result_dirs, args.out_dir)
    figure_market_prediction_curve(args.data_dir, result_dirs, args.out_dir, args.seq_len, args.pred_len)
    figure_rolling_direction_accuracy(args.data_dir, result_dirs, args.out_dir, args.seq_len, args.pred_len)
    figure_single_ticker_prediction_examples(args.data_dir, result_dirs, args.out_dir, args.seq_len, args.pred_len)
    figure_validation_direction_curve(result_dirs, args.out_dir)
    print(f"Saved S&P500 figures to {args.out_dir}")


if __name__ == "__main__":
    main()

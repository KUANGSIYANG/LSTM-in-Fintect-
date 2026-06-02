from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import patches
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from fintech_transfer_data import ForecastWindowDataset, load_series_csv, pchange_to_class, split_dataset
from run_pslstm_transfer import (
    build_model,
    collect_target_predictions,
    load_compatible_state_dict,
    make_loader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"
LANG = "en"
PNG_DPI = 150
OUTPUT_FORMATS = ("png", "pdf", "svg")

COLORS = {
    "blue": "#3B6FB6",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#7B4FA3",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
}


ZH = {
    "Target-only": "仅目标域",
    "Weather": "Weather",
    "Weather-6": "Weather-6",
    "ETTm1": "ETTm1",
    "Electricity": "Electricity",
    "Directional Accuracy": "方向准确率",
    "6-Class Accuracy": "6分类准确率",
    "Macro F1": "Macro F1",
    "LSTM target-only": "LSTM 仅目标域",
    "P-sLSTM target-only": "P-sLSTM 仅目标域",
    "P-sLSTM Weather fine-tune": "P-sLSTM Weather微调",
    "Non-financial\nsources": "非金融\n源域数据",
    "HS300\nfinancial data": "沪深300\n金融数据",
    "P-sLSTM\npretraining": "P-sLSTM\n源域预训练",
    "Target\nfine-tuning": "目标域\n微调",
    "Prediction &\nevaluation": "预测与\n评估",
    "Weather / ETTm1 / Electricity": "气象 / ETTm1 / 电力",
    "Metrics: MAE, RMSE, directional accuracy,\n6-class accuracy, macro F1": "指标：MAE、RMSE、方向准确率、\n6分类准确率、Macro F1",
    "(a) Experimental framework": "(a) 实验框架",
    "(a) Target-domain training": "(a) 目标域训练",
    "(b) Source-domain pretraining": "(b) 源域预训练",
    "Epoch": "训练轮次",
    "Validation loss": "验证损失",
    "Same-distribution holdout metrics": "同分布验证指标",
    "Out-of-time 2019 metrics": "2019年时间外推测试指标",
    "Score": "分数",
    "lower is better": "越低越好",
    "Prediction trajectory": "预测轨迹",
    "Predicted vs. true": "预测值与真实值",
    "Validation samples": "验证样本",
    "p_change (%)": "涨跌幅（%）",
    "True": "真实值",
    "Predicted": "预测值",
    "True p_change (%)": "真实涨跌幅（%）",
    "Predicted p_change (%)": "预测涨跌幅（%）",
    "(a) LSTM target-only": "(a) LSTM 仅目标域",
    "(b) P-sLSTM Weather fine-tune": "(b) P-sLSTM Weather微调",
    "Gain over P-sLSTM target-only": "相对 P-sLSTM 仅目标域增益",
    "Transfer gain under same-distribution holdout": "同分布验证下的迁移增益",
}


def tr(label: str) -> str:
    return ZH.get(label, label) if LANG == "zh" else label


SAME_DIST = pd.DataFrame(
    [
        ["LSTM", "Target-only", 1.5135, 2.1844, 0.5060, 0.2478, 0.1180],
        ["P-sLSTM", "Target-only", 1.4959, 2.1580, 0.5464, 0.2586, 0.1529],
        ["P-sLSTM", "Weather", 1.4999, 2.1569, 0.5568, 0.2671, 0.1569],
        ["P-sLSTM", "Weather-6", 1.4912, 2.1490, 0.5563, 0.2681, 0.1538],
        ["P-sLSTM", "ETTm1", 1.4982, 2.1585, 0.5517, 0.2636, 0.1535],
        ["P-sLSTM", "Electricity", 1.5036, 2.1628, 0.5420, 0.2564, 0.1527],
    ],
    columns=["Model", "Strategy", "MAE", "RMSE", "Directional Accuracy", "6-Class Accuracy", "Macro F1"],
)


OUT_TIME = pd.DataFrame(
    [
        ["LSTM", "Target-only", 2.1364, 2.9885, 0.4781, 0.1769, 0.0907],
        ["LSTM", "Weather", 2.1652, 3.0196, 0.4673, 0.1867, 0.0976],
        ["P-sLSTM", "Target-only", 2.1918, 3.0532, 0.5132, 0.1825, 0.1150],
        ["P-sLSTM", "Weather", 2.2073, 3.0851, 0.4937, 0.1874, 0.1205],
        ["P-sLSTM", "Weather-6", 2.1800, 3.0441, 0.5016, 0.1877, 0.1176],
        ["P-sLSTM", "ETTm1", 2.2077, 3.0819, 0.4879, 0.1861, 0.1184],
        ["P-sLSTM", "Electricity", 2.1824, 3.0260, 0.5069, 0.1781, 0.1080],
    ],
    columns=["Model", "Strategy", "MAE", "RMSE", "Directional Accuracy", "6-Class Accuracy", "Macro F1"],
)


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
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in OUTPUT_FORMATS:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = PNG_DPI
        fig.savefig(FIG_DIR / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def load_summary(name: str):
    path = PROJECT_ROOT / "outputs" / name / "run_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def figure_framework():
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    boxes = [
        (0.2, 2.4, 1.7, 0.8, tr("Non-financial\nsources"), COLORS["blue"]),
        (0.2, 0.8, 1.7, 0.8, tr("HS300\nfinancial data"), COLORS["orange"]),
        (2.6, 2.4, 1.8, 0.8, tr("P-sLSTM\npretraining"), COLORS["blue"]),
        (5.0, 1.6, 1.8, 0.8, tr("Target\nfine-tuning"), COLORS["green"]),
        (7.4, 1.6, 2.0, 0.8, tr("Prediction &\nevaluation"), COLORS["purple"]),
    ]
    for x, y, w, h, text, color in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color + "18",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", weight="bold", color="#111827")

    arrow_kw = dict(arrowstyle="->", mutation_scale=12, linewidth=1.5, color="#374151")
    ax.annotate("", xy=(2.55, 2.8), xytext=(1.95, 2.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.0, 2.0), xytext=(4.45, 2.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.0, 2.0), xytext=(1.95, 1.2), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.35, 2.0), xytext=(6.85, 2.0), arrowprops=arrow_kw)

    ax.text(1.05, 3.45, tr("Weather / ETTm1 / Electricity"), ha="center", color=COLORS["blue"])
    ax.text(5.9, 0.8, tr("Metrics: MAE, RMSE, directional accuracy,\n6-class accuracy, macro F1"), ha="center", color=COLORS["gray"])
    ax.text(0.0, 3.85, tr("(a) Experimental framework"), weight="bold", ha="left")
    save_figure(fig, "Fig1_framework")


def plot_histories(ax, histories, title):
    for label, history, color in histories:
        epochs = [item["epoch"] for item in history]
        train = [item["train_loss"] for item in history]
        val = [item["val_loss"] for item in history]
        ax.plot(epochs, train, color=color, alpha=0.35, linewidth=1.5, linestyle="--")
        ax.plot(epochs, val, color=color, linewidth=2.0, label=tr(label))
    ax.set_title(tr(title))
    ax.set_xlabel(tr("Epoch"))
    ax.set_ylabel(tr("Validation loss"))
    ax.legend(frameon=False)


def figure_training_curves():
    pslstm = load_summary("weather_to_hs300_1718_to_19_pslstm")
    lstm = load_summary("weather_to_hs300_1718_to_19_lstm")
    weather6 = load_summary("weather6_to_hs300_1718_to_19_pslstm")
    ettm1 = load_summary("ettm1_to_hs300_1718_to_19_pslstm")
    electricity = load_summary("electricity_to_hs300_1718_to_19_pslstm")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), sharey=False)
    plot_histories(
        axes[0],
        [
            ("LSTM target-only", lstm["baseline"]["history"], COLORS["gray"]),
            ("P-sLSTM target-only", pslstm["baseline"]["history"], COLORS["blue"]),
            ("P-sLSTM Weather fine-tune", pslstm["finetune"]["history"], COLORS["green"]),
        ],
        "(a) Target-domain training",
    )
    plot_histories(
        axes[1],
        [
            ("Weather-6", weather6["pretrain"]["history"], COLORS["green"]),
            ("ETTm1", ettm1["pretrain"]["history"], COLORS["blue"]),
            ("Electricity", electricity["pretrain"]["history"], COLORS["orange"]),
        ],
        "(b) Source-domain pretraining",
    )
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "Fig2_training_curves")


def grouped_metric_bars(df, metrics, title, name, ylim=None):
    labels = [f"{m}\n{tr(s)}" for m, s in zip(df["Model"], df["Strategy"])]
    x = np.arange(len(df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    palette = [COLORS["blue"], COLORS["green"], COLORS["orange"]]
    for i, metric in enumerate(metrics):
        values = df[metric].to_numpy()
        ax.bar(x + (i - 1) * width, values, width=width, label=tr(metric), color=palette[i], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(tr(title))
    ax.set_ylabel(tr("Score"))
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=len(metrics), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    save_figure(fig, name)


def figure_metrics():
    grouped_metric_bars(
        SAME_DIST,
        ["Directional Accuracy", "6-Class Accuracy", "Macro F1"],
        "Same-distribution holdout metrics",
        "Fig3_same_distribution_metrics",
        ylim=(0.0, 0.62),
    )
    grouped_metric_bars(
        OUT_TIME,
        ["Directional Accuracy", "6-Class Accuracy", "Macro F1"],
        "Out-of-time 2019 metrics",
        "Fig4_out_of_time_metrics",
        ylim=(0.0, 0.56),
    )


def figure_regression_metrics():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for ax, metric, better in [(axes[0], "MAE", "lower is better"), (axes[1], "RMSE", "lower is better")]:
        rows = SAME_DIST[SAME_DIST["Model"].eq("P-sLSTM")].copy()
        x = np.arange(len(rows))
        ax.bar(x, rows[metric], color=[COLORS["gray"], COLORS["green"], COLORS["green"], COLORS["blue"], COLORS["orange"]])
        ax.set_xticks(x)
        ax.set_xticklabels([tr(item) for item in rows["Strategy"]], rotation=25, ha="right")
        ax.set_title(f"{tr(metric)} 不同迁移源对比" if LANG == "zh" else f"{metric} across transfer sources")
        ax.set_ylabel(metric)
        ax.text(0.02, 0.95, tr(better), transform=ax.transAxes, va="top", color=COLORS["gray"])
    fig.tight_layout()
    save_figure(fig, "Fig5_source_regression_metrics")


def build_eval_args(model_name: str):
    return SimpleNamespace(
        seq_len=20,
        pred_len=1,
        target_col="p_change",
        max_features=0,
        train_ratio=0.8,
        seed=2026,
        batch_size=512,
        model=model_name,
        embedding_dim=64,
        patch_size=5,
        stride=2,
        num_heads=4,
        conv1d_kernel_size=2,
        num_blocks=1,
        num_layers=1,
        backend="vanilla",
        direction_head=False,
        last_value_residual=False,
    )


def collect_holdout_predictions(model_name: str, ckpt_path: Path):
    args = build_eval_args(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    series = load_series_csv(PROJECT_ROOT / "data" / "train_mix-17-18.csv", target_col="p_change", max_features=0)
    dataset = ForecastWindowDataset(series, args.seq_len, args.pred_len)
    _, val_set = split_dataset(dataset, args.train_ratio, args.seed)
    loader = make_loader(val_set, args, shuffle=False)
    model = build_model(args, channel=len(series.feature_cols)).to(device)
    load_compatible_state_dict(model, ckpt_path, device)
    pred, true, _ = collect_target_predictions(model, loader, series, device)
    return true, pred


def figure_prediction_quality():
    best_ckpt = PROJECT_ROOT / "outputs" / "weather_to_hs300_1718_to_19_pslstm" / "P_sLSTM_finetuned.pt"
    true, pred = collect_holdout_predictions("P_sLSTM", best_ckpt)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    n = min(240, len(true))
    axes[0].plot(true[:n], color=COLORS["gray"], linewidth=1.5, label=tr("True"))
    axes[0].plot(pred[:n], color=COLORS["blue"], linewidth=1.2, label=tr("Predicted"))
    axes[0].axhline(0, color="#111827", linewidth=0.8, alpha=0.6)
    axes[0].set_title(f"(a) {tr('Prediction trajectory')}")
    axes[0].set_xlabel(tr("Validation samples"))
    axes[0].set_ylabel(tr("p_change (%)"))
    axes[0].legend(frameon=False)

    axes[1].scatter(true, pred, s=7, alpha=0.22, color=COLORS["blue"], edgecolor="none")
    lim = np.percentile(np.abs(np.concatenate([true, pred])), 98)
    lim = max(lim, 2.0)
    axes[1].plot([-lim, lim], [-lim, lim], color=COLORS["red"], linewidth=1.0, linestyle="--")
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_title(f"(b) {tr('Predicted vs. true')}")
    axes[1].set_xlabel(tr("True p_change (%)"))
    axes[1].set_ylabel(tr("Predicted p_change (%)"))
    fig.tight_layout()
    save_figure(fig, "Fig6_prediction_quality")

    return true, pred


def figure_confusion_matrices():
    lstm_true, lstm_pred = collect_holdout_predictions(
        "LSTM", PROJECT_ROOT / "outputs" / "weather_to_hs300_1718_to_19_lstm" / "LSTM_baseline.pt"
    )
    pslstm_true, pslstm_pred = collect_holdout_predictions(
        "P_sLSTM", PROJECT_ROOT / "outputs" / "weather_to_hs300_1718_to_19_pslstm" / "P_sLSTM_finetuned.pt"
    )

    class_names = ["<=-2", "-2~-1", "-1~0", "0~1", "1~2", ">2"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for ax, title, true, pred in [
        (axes[0], "(a) LSTM target-only", lstm_true, lstm_pred),
        (axes[1], "(b) P-sLSTM Weather fine-tune", pslstm_true, pslstm_pred),
    ]:
        cm = confusion_matrix(pchange_to_class(true), pchange_to_class(pred), labels=np.arange(6), normalize="true")
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
        ax.set_title(tr(title))
        ax.grid(False)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_figure(fig, "Fig7_confusion_matrices")


def figure_transfer_gain():
    rows = SAME_DIST.copy()
    baseline = rows[(rows["Model"] == "P-sLSTM") & (rows["Strategy"] == "Target-only")].iloc[0]
    transfer = rows[(rows["Model"] == "P-sLSTM") & (rows["Strategy"] != "Target-only")].copy()
    metrics = ["Directional Accuracy", "6-Class Accuracy", "Macro F1"]
    gains = transfer[metrics].to_numpy() - baseline[metrics].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(transfer))
    width = 0.23
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, gains[:, i], width=width, label=tr(metric), color=[COLORS["blue"], COLORS["green"], COLORS["orange"]][i])
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([tr(s) for s in transfer["Strategy"]], rotation=15, ha="right")
    ax.set_ylabel(tr("Gain over P-sLSTM target-only"))
    ax.set_title(tr("Transfer gain under same-distribution holdout"))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    save_figure(fig, "Fig8_transfer_gain")


def main():
    global FIG_DIR, OUTPUT_FORMATS
    parser = argparse.ArgumentParser(description="Draw paper figures for the HS300 transfer experiment.")
    parser.add_argument("--out_dir", type=Path, default=FIG_DIR)
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--png_dpi", type=int, default=150)
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=None)
    args = parser.parse_args()

    FIG_DIR = args.out_dir
    OUTPUT_FORMATS = tuple(args.formats or (["png"] if args.lang == "zh" else ["png", "pdf", "svg"]))

    setup_style(args.lang, args.png_dpi)
    figure_framework()
    figure_training_curves()
    figure_metrics()
    figure_regression_metrics()
    figure_prediction_quality()
    figure_confusion_matrices()
    figure_transfer_gain()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

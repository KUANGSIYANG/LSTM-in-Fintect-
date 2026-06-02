from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from fintech_transfer_data import ForecastWindowDataset, load_series_csv, pchange_to_class, split_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "P-sLSTM"
sys.path.insert(0, str(VENDOR_ROOT))

from models import LSTM, P_sLSTM  # noqa: E402


class LastValueResidualWrapper(nn.Module):
    """NLinear-style residual wrapper: predict a delta from the latest input step."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        last_value = x[:, -1:, :]
        residual = self.base_model(x - last_value)
        if residual.size(1) != 1:
            last_value = last_value.repeat(1, residual.size(1), 1)
        return residual + last_value


class DirectionHeadWrapper(nn.Module):
    """Auxiliary direction classifier on the target-channel P-sLSTM representation."""

    def __init__(self, base_model: nn.Module, target_idx: int, feature_dim: int):
        super().__init__()
        self.base_model = base_model
        self.target_idx = target_idx
        self.direction_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 1),
        )
        self._last_direction_logits = None

    def forward(self, x):
        pred = self.base_model(x)
        features = getattr(self.base_model, "_last_channel_features", None)
        if features is None:
            raise RuntimeError("direction_head requires a model that exposes _last_channel_features")
        logits = self.direction_head(features[:, self.target_idx, :]).squeeze(-1)
        self._last_direction_logits = logits
        return pred


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args, channel: int, target_idx: int | None = None):
    config = SimpleNamespace(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        channel=channel,
        embedding_dim=args.embedding_dim,
        patch_size=args.patch_size,
        stride=args.stride,
        num_heads=args.num_heads,
        conv1d_kernel_size=args.conv1d_kernel_size,
        num_blocks=args.num_blocks,
        num_layers=args.num_layers,
        backend=args.backend,
    )

    if args.model == "P_sLSTM":
        model = P_sLSTM.Model(config)
        if args.direction_head:
            if target_idx is None:
                raise ValueError("target_idx is required when --direction_head is enabled")
            feature_dim = model.embedding_dim * model.patch_num
            model = DirectionHeadWrapper(model, target_idx=target_idx, feature_dim=feature_dim)
    elif args.model == "LSTM":
        model = LSTM.Model(config)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.last_value_residual:
        model = LastValueResidualWrapper(model)
    return model


def load_compatible_state_dict(model, checkpoint_path: Path, device):
    source_state = torch.load(checkpoint_path, map_location=device)
    target_state = model.state_dict()
    compatible_state = {}
    skipped = []

    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped.append(key)

    target_state.update(compatible_state)
    model.load_state_dict(target_state)
    print(f"loaded {len(compatible_state)} tensors from {checkpoint_path}")
    if skipped:
        print(f"skipped {len(skipped)} incompatible tensors")


def make_loader(dataset, args, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def binary_focal_loss_with_logits(logits, labels, alpha: float, gamma: float):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * labels + (1.0 - prob) * (1.0 - labels)
    alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def compute_loss(
    pred,
    y,
    criterion,
    target_idx: int | None,
    args=None,
    raw_target=None,
    target_zero_norm: float | None = None,
    direction_logits=None,
):
    if target_idx is None:
        reg_loss = criterion(pred, y)
    else:
        reg_loss = criterion(pred[:, :, target_idx], y[:, :, target_idx])

    if (
        args is None
        or getattr(args, "direction_loss_weight", 0.0) <= 0
        or target_idx is None
        or raw_target is None
        or target_zero_norm is None
    ):
        return reg_loss

    if getattr(args, "direction_head", False) and direction_logits is not None:
        direction_labels = (raw_target[:, -1] > 0).float().to(direction_logits.device)
        direction_logits = direction_logits * args.direction_logit_scale
    else:
        direction_logits = (pred[:, :, target_idx] - target_zero_norm) * args.direction_logit_scale
        direction_labels = (raw_target > 0).float().to(direction_logits.device)
    if args.direction_loss == "focal":
        direction_loss = binary_focal_loss_with_logits(
            direction_logits,
            direction_labels,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )
    else:
        direction_loss = F.binary_cross_entropy_with_logits(direction_logits, direction_labels)
    return reg_loss + args.direction_loss_weight * direction_loss


def run_epoch(model, loader, criterion, optimizer, device, target_idx: int | None, args, target_zero_norm: float | None):
    model.train()
    total_loss = 0.0
    total_count = 0

    for x, y, raw_target in loader:
        x = x.to(device)
        y = y.to(device)
        raw_target = raw_target.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        direction_logits = getattr(model, "_last_direction_logits", None)
        loss = compute_loss(pred, y, criterion, target_idx, args, raw_target, target_zero_norm, direction_logits)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, args, target_idx: int | None, target_zero_norm: float | None):
    model.eval()
    total_loss = 0.0
    total_count = 0
    for x, y, raw_target in loader:
        x = x.to(device)
        y = y.to(device)
        raw_target = raw_target.to(device)
        pred = model(x)
        direction_logits = getattr(model, "_last_direction_logits", None)
        loss = compute_loss(pred, y, criterion, target_idx, args, raw_target, target_zero_norm, direction_logits)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_direction_on_normalized_output(model, loader, device, target_idx: int, target_zero_norm: float):
    model.eval()
    true_up = []
    pred_up = []
    for x, _, raw_target in loader:
        x = x.to(device)
        pred = model(x)
        direction_logits = getattr(model, "_last_direction_logits", None)
        if direction_logits is not None:
            pred_target = direction_logits.cpu().numpy().reshape(-1)
            threshold = 0.0
        else:
            pred_target = pred[:, :, target_idx].cpu().numpy().reshape(-1)
            threshold = target_zero_norm
        raw = raw_target.numpy().reshape(-1)
        true_up.extend((raw > 0).tolist())
        pred_up.extend((pred_target > threshold).tolist())
    return {
        "val_directional_accuracy": float(accuracy_score(true_up, pred_up)),
        "val_binary_f1": float(f1_score(true_up, pred_up, zero_division=0)),
    }


def train_model(
    model,
    train_loader,
    val_loader,
    args,
    device,
    checkpoint_path: Path,
    target_idx: int,
    target_zero_norm: float,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    maximize_selection = args.selection_metric in {"direction_acc", "binary_f1"}
    best_selection = -float("inf") if maximize_selection else float("inf")
    best_val_at_selection = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss_target_idx = target_idx if args.target_only_loss else None
        model._loss_target_idx = loss_target_idx
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, loss_target_idx, args, target_zero_norm)
        val_loss = evaluate_loss(model, val_loader, criterion, device, args, loss_target_idx, target_zero_norm)
        val_direction = evaluate_direction_on_normalized_output(model, val_loader, device, target_idx, target_zero_norm)
        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_direction}
        history.append(entry)

        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
            f"| val_dir_acc={val_direction['val_directional_accuracy']:.6f} "
            f"| val_bin_f1={val_direction['val_binary_f1']:.6f}"
        )

        if args.selection_metric == "direction_acc":
            selection_value = val_direction["val_directional_accuracy"]
        elif args.selection_metric == "binary_f1":
            selection_value = val_direction["val_binary_f1"]
        else:
            selection_value = val_loss
        improved = selection_value > best_selection if maximize_selection else selection_value < best_selection

        if improved:
            best_selection = selection_value
            best_val_at_selection = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}; best epoch was {best_epoch}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_at_selection,
        "selection_metric": args.selection_metric,
        "best_selection_value": float(best_selection),
        "history": history,
    }


@torch.no_grad()
def collect_target_predictions(model, loader, series, device, use_direction_head: bool = False):
    model.eval()
    pred_values = []
    true_values = []
    direction_scores = []

    target_idx = series.target_idx
    target_mean = float(series.mean[target_idx])
    target_std = float(series.std[target_idx])

    for x, y, raw_target in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        pred_target = pred[:, -1, target_idx] * target_std + target_mean
        true_target = raw_target[:, -1].numpy()
        pred_values.extend(pred_target.tolist())
        true_values.extend(true_target.tolist())
        logits = getattr(model, "_last_direction_logits", None)
        if use_direction_head and logits is not None:
            direction_scores.extend(logits.cpu().numpy().tolist())

    if use_direction_head and direction_scores:
        return np.asarray(pred_values), np.asarray(true_values), np.asarray(direction_scores)
    return np.asarray(pred_values), np.asarray(true_values), None


def score_prediction(true_values: np.ndarray, pred_values: np.ndarray, metric: str) -> float:
    pred_classes = pchange_to_class(pred_values)
    true_classes = pchange_to_class(true_values)
    if metric == "direction":
        return float(accuracy_score(true_values > 0, pred_values > 0))
    if metric == "class_accuracy":
        return float(accuracy_score(true_classes, pred_classes))
    if metric == "macro_f1":
        return float(f1_score(true_classes, pred_classes, average="macro", zero_division=0))
    raise ValueError(f"Unsupported calibration metric: {metric}")


def calibrate_affine(true_values: np.ndarray, pred_values: np.ndarray, metric: str):
    best = {"score": -1.0, "scale": 1.0, "bias": 0.0}
    for scale in np.linspace(0.7, 1.3, 25):
        scaled = pred_values * scale
        for bias in np.linspace(-0.8, 0.8, 65):
            score = score_prediction(true_values, scaled + bias, metric)
            if score > best["score"]:
                best = {"score": score, "scale": float(scale), "bias": float(bias)}
    return best


@torch.no_grad()
def evaluate_finance_metrics(
    model,
    loader,
    series,
    device,
    output_csv: Path | None = None,
    scale: float = 1.0,
    bias: float = 0.0,
    use_direction_head: bool = False,
):
    pred_values, true_values, direction_scores = collect_target_predictions(model, loader, series, device, use_direction_head)
    direction_values = direction_scores if direction_scores is not None else pred_values
    direction_values = direction_values * scale + bias
    pred_classes = pchange_to_class(pred_values)
    true_classes = pchange_to_class(true_values)

    metrics = {
        "mae": float(mean_absolute_error(true_values, pred_values)),
        "rmse": float(mean_squared_error(true_values, pred_values, squared=False)),
        "directional_accuracy": float(accuracy_score(true_values > 0, direction_values > 0)),
        "binary_f1": float(f1_score(true_values > 0, direction_values > 0, zero_division=0)),
        "class_accuracy": float(accuracy_score(true_classes, pred_classes)),
        "macro_f1": float(f1_score(true_classes, pred_classes, average="macro", zero_division=0)),
        "samples": int(len(true_values)),
    }
    if len(np.unique(true_values > 0)) == 2:
        metrics["direction_auc"] = float(roc_auc_score(true_values > 0, direction_values))
    else:
        metrics["direction_auc"] = None

    report = classification_report(true_classes, pred_classes, zero_division=0)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(report)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if direction_scores is not None:
            rows = np.column_stack([true_values, pred_values, direction_values, true_classes, pred_classes])
            header = "true_p_change,pred_p_change,direction_score,true_class,pred_class"
        else:
            rows = np.column_stack([true_values, pred_values, true_classes, pred_classes])
            header = "true_p_change,pred_p_change,true_class,pred_class"
        np.savetxt(
            output_csv,
            rows,
            delimiter=",",
            header=header,
            comments="",
        )

    return metrics, report


def prepare_dataset(csv_path: Path, args, fit_stats=True, mean=None, std=None, feature_cols=None):
    series = load_series_csv(
        csv_path,
        target_col=args.target_col,
        feature_cols=feature_cols,
        max_features=args.max_features,
        fit_stats=fit_stats,
        mean=mean,
        std=std,
    )
    dataset = ForecastWindowDataset(series, seq_len=args.seq_len, pred_len=args.pred_len)
    if args.max_windows > 0 and len(dataset) > args.max_windows:
        dataset = Subset(dataset, range(args.max_windows))
    return series, dataset


def train_stage(stage_name: str, csv_path: Path, args, device, load_checkpoint: Path | None, save_checkpoint: Path):
    print(f"\n===== {stage_name}: {csv_path} =====")
    series, dataset = prepare_dataset(csv_path, args)
    train_set, val_set = split_dataset(dataset, args.train_ratio, args.seed)
    train_loader = make_loader(train_set, args, shuffle=True)
    val_loader = make_loader(val_set, args, shuffle=False)

    model = build_model(args, channel=len(series.feature_cols), target_idx=series.target_idx).to(device)
    if load_checkpoint is not None and load_checkpoint.exists():
        load_compatible_state_dict(model, load_checkpoint, device)

    target_zero_norm = float((0.0 - series.mean[series.target_idx]) / series.std[series.target_idx])
    info = train_model(model, train_loader, val_loader, args, device, save_checkpoint, series.target_idx, target_zero_norm)
    return series, model, info


def main():
    parser = argparse.ArgumentParser(description="P-sLSTM transfer learning for fintech prediction")
    parser.add_argument("--mode", choices=["baseline", "pretrain", "finetune", "evaluate", "all"], default="all")
    parser.add_argument("--model", choices=["P_sLSTM", "LSTM"], default="P_sLSTM")
    parser.add_argument("--source_csv", type=Path, default=PROJECT_ROOT / "data" / "train_mix-17-18.csv")
    parser.add_argument("--target_train_csv", type=Path, default=PROJECT_ROOT / "data" / "train_mix-19.csv")
    parser.add_argument("--target_test_csv", type=Path, default=PROJECT_ROOT / "data" / "test_mix.csv")
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "outputs" / "pslstm_transfer")
    parser.add_argument("--target_col", default="p_change")
    parser.add_argument("--max_features", type=int, default=0, help="limit numeric features; 0 means all features")
    parser.add_argument("--target_only_loss", action="store_true", help="optimize only the target channel")
    parser.add_argument("--last_value_residual", action="store_true", help="predict residual from the latest observed value")
    parser.add_argument("--direction_loss_weight", type=float, default=0.0, help="add direction-aware BCE/focal loss on target sign")
    parser.add_argument("--direction_loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--direction_logit_scale", type=float, default=1.0)
    parser.add_argument("--direction_head", action="store_true", help="add an auxiliary direction classification head")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--selection_metric", choices=["val_loss", "direction_acc", "binary_f1"], default="val_loss")
    parser.add_argument("--calibrate", action="store_true", help="fit affine prediction calibration on target validation split")
    parser.add_argument("--calibration_metric", choices=["macro_f1", "class_accuracy", "direction"], default="macro_f1")

    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--conv1d_kernel_size", type=int, default=2)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--backend", choices=["vanilla", "cuda"], default="vanilla")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_windows", type=int, default=0, help="limit windows for smoke tests; 0 means full data")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"device: {device}")

    source_ckpt = args.out_dir / f"{args.model}_source_pretrained.pt"
    finetuned_ckpt = args.out_dir / f"{args.model}_finetuned.pt"
    baseline_ckpt = args.out_dir / f"{args.model}_baseline.pt"

    summary_path = args.out_dir / "run_summary.json"
    if summary_path.exists():
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        run_summary = {}

    if args.mode in {"baseline", "all"}:
        _, _, info = train_stage("baseline target training", args.target_train_csv, args, device, None, baseline_ckpt)
        run_summary["baseline"] = info

    if args.mode in {"pretrain", "all"}:
        _, _, info = train_stage("source pretraining", args.source_csv, args, device, None, source_ckpt)
        run_summary["pretrain"] = info

    if args.mode in {"finetune", "all"}:
        load_ckpt = source_ckpt if source_ckpt.exists() else None
        _, _, info = train_stage("target finetuning", args.target_train_csv, args, device, load_ckpt, finetuned_ckpt)
        run_summary["finetune"] = info

    if args.mode in {"evaluate", "all"}:
        target_ref_series, _ = prepare_dataset(args.target_train_csv, args)
        _, target_full_dataset = prepare_dataset(args.target_train_csv, args)
        _, target_val_set = split_dataset(target_full_dataset, args.train_ratio, args.seed)
        target_val_loader = make_loader(target_val_set, args, shuffle=False)
        test_series, test_dataset = prepare_dataset(
            args.target_test_csv,
            args,
            fit_stats=False,
            mean=target_ref_series.mean,
            std=target_ref_series.std,
            feature_cols=target_ref_series.feature_cols,
        )
        test_loader = make_loader(test_dataset, args, shuffle=False)

        for name, ckpt in [("baseline", baseline_ckpt), ("finetuned", finetuned_ckpt)]:
            if not ckpt.exists():
                continue
            print(f"\n===== evaluate {name}: {ckpt} =====")
            model = build_model(args, channel=len(test_series.feature_cols), target_idx=test_series.target_idx).to(device)
            load_compatible_state_dict(model, ckpt, device)
            calibration = {"scale": 1.0, "bias": 0.0, "score": None}
            if args.calibrate:
                val_pred, val_true, val_direction = collect_target_predictions(
                    model,
                    target_val_loader,
                    target_ref_series,
                    device,
                    use_direction_head=args.direction_head,
                )
                calibration_scores = val_direction if val_direction is not None else val_pred
                calibration = calibrate_affine(val_true, calibration_scores, args.calibration_metric)
                print(f"calibration: {calibration}")
            metrics, report = evaluate_finance_metrics(
                model,
                test_loader,
                test_series,
                device,
                output_csv=args.out_dir / f"{name}_predictions.csv",
                scale=calibration["scale"],
                bias=calibration["bias"],
                use_direction_head=args.direction_head,
            )
            metrics["calibration"] = calibration
            run_summary[f"{name}_metrics"] = metrics
            (args.out_dir / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")

    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Metric computation utilities for supervised evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        cm[int(y_true), int(y_pred)] += 1
    return cm


def save_confusion_plot(
    cm: np.ndarray,
    class_names: Iterable[str],
    save_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping confusion matrix plot.")
        return

    num_classes = cm.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, where=row_sums != 0)
        normalized[row_sums.squeeze(-1) == 0] = 0.0

    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(6, num_classes)))
    im = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax)
    display_names = list(class_names)[:num_classes]
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=display_names,
        yticklabels=display_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(num_classes):
        for j in range(num_classes):
            value = normalized[i, j]
            color = "white" if value > 0.5 else "black"
            ax.text(j, i, f"{100 * value:.1f}%", ha="center", va="center", color=color)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path}")


def save_predictions_csv(
    labels: np.ndarray,
    preds: np.ndarray,
    meta_df: Optional[pd.DataFrame],
    save_path: Path,
    extra_fields: Optional[dict[str, object]] = None,
    probs: Optional[np.ndarray] = None,
) -> None:
    df = pd.DataFrame({"true_label": labels, "pred_label": preds})
    if probs is not None:
        for c in range(probs.shape[1]):
            df[f"prob_class_{c}"] = probs[:, c]
    if extra_fields:
        for key, value in extra_fields.items():
            df.insert(0, key, value)
    if meta_df is not None:
        df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved predictions to {save_path}")


def compute_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute AUROC. Binary: uses positive-class probability. Multiclass: macro OvR."""
    if labels.size == 0 or probs is None or probs.ndim < 2:
        return float("nan")
    n_classes = probs.shape[1]
    try:
        if n_classes == 2:
            return float(roc_auc_score(labels, probs[:, 1]))
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def compute_auprc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute AUPRC (average precision). Binary: positive-class score. Multiclass: macro OvR."""
    if labels.size == 0 or probs is None or probs.ndim < 2:
        return float("nan")
    n_classes = probs.shape[1]
    try:
        if n_classes == 2:
            return float(average_precision_score(labels, probs[:, 1]))
        aps = []
        for c in range(n_classes):
            binary = (labels == c).astype(int)
            if binary.sum() == 0 or binary.sum() == len(binary):
                continue
            aps.append(float(average_precision_score(binary, probs[:, c])))
        return float(np.mean(aps)) if aps else float("nan")
    except Exception:
        return float("nan")


def regression_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    target_names: Optional[list] = None,
) -> dict[str, float]:
    """Compute unit_polar regression metrics.

    Inputs: targets [N, 2] = [v_radial, v_lateral]; preds [N, 3] = [a, b, speed].
    """
    del target_names  # kept for API compatibility; unit_polar metric names are fixed.
    if targets.size == 0:
        return {"mae": 0.0, "rmse": 0.0}

    eps = 1e-8
    v_r, v_l = targets[:, 0], targets[:, 1]
    a_p, b_p, speed_p = preds[:, 0], preds[:, 1], preds[:, 2]
    norm_p = np.sqrt(a_p ** 2 + b_p ** 2 + eps)
    a_p_unit = a_p / norm_p
    b_p_unit = b_p / norm_p
    speed_t = np.sqrt(v_r ** 2 + v_l ** 2)
    speed_err = np.abs(speed_p - speed_t)
    angle_true = np.degrees(np.arctan2(v_l, v_r))
    angle_pred = np.degrees(np.arctan2(b_p_unit, a_p_unit))
    angle_diff = (angle_pred - angle_true + 180) % 360 - 180
    vx_p = a_p_unit * speed_p
    vy_p = b_p_unit * speed_p
    vec_err = np.sqrt((vx_p - v_r) ** 2 + (vy_p - v_l) ** 2)
    mae_speed = float(speed_err.mean())
    rmse_speed = float(np.sqrt((speed_err ** 2).mean()))
    mae_radial = float(np.abs(vx_p - v_r).mean())
    mae_lateral = float(np.abs(vy_p - v_l).mean())
    return {
        "mae_speed": mae_speed,
        "rmse_speed": rmse_speed,
        "mae_angle_deg": float(np.abs(angle_diff).mean()),
        "mae_vec": float(vec_err.mean()),
        "mae_radial": mae_radial,
        "mae_lateral": mae_lateral,
        "mae": mae_speed,
        "rmse": rmse_speed,
    }


def save_regression_csv(
    targets: np.ndarray,
    preds: np.ndarray,
    target_names: list,
    meta_df: Optional[pd.DataFrame],
    save_path: Path,
    extra_fields: Optional[dict] = None,
) -> None:
    if targets.ndim == 1:
        targets = targets[:, None]
        preds = preds[:, None]
    rows: dict[str, np.ndarray] = {}
    for i, name in enumerate(target_names):
        rows[f"true_{name}"] = targets[:, i]
        rows[f"pred_{name}"] = preds[:, i]
    df = pd.DataFrame(rows)
    if extra_fields:
        for key, value in extra_fields.items():
            df.insert(0, key, value)
    if meta_df is not None:
        df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved regression predictions to {save_path}")


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    if labels.size == 0:
        return {
            "balanced_acc": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
        }
    accuracy = float((preds == labels).mean())
    return {
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "accuracy": accuracy,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }

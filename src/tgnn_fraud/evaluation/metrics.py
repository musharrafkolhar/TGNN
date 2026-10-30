"""Shared evaluation with strict input checks and validation-only thresholding."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

REQUIRED = {"edge_id", "split", "score", "label", "model", "run_id"}


def validate_predictions(frame: pd.DataFrame) -> None:
    """Reject invalid, mixed-run or ambiguous prediction files."""
    if not REQUIRED.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing prediction columns or empty file")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Prediction fields must not be missing")
    for name in ["edge_id", "score"]:
        if not pd.api.types.is_numeric_dtype(frame[name]) or not np.isfinite(frame[name]).all():
            raise ValueError(f"{name} must contain finite numeric values")
    for name in ["model", "run_id"]:
        if frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} must not be blank")
    if frame.edge_id.duplicated().any() or (frame.edge_id < 0).any():
        raise ValueError("edge_id must be unique and nonnegative within a model run")
    if not np.equal(frame.edge_id, np.floor(frame.edge_id)).all():
        raise ValueError("edge_id must contain integers")
    if frame.model.nunique() != 1 or frame.run_id.nunique() != 1:
        raise ValueError("Evaluate one model run at a time")
    if not frame.split.isin(["train", "val", "test"]).all():
        raise ValueError("Unknown split name")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    if not np.isfinite(frame.score).all() or not frame.score.between(0, 1).all():
        raise ValueError("Scores must be finite in [0,1]")
    for split in ["val", "test"]:
        subset = frame.loc[frame.split == split]
        if subset.empty:
            raise ValueError(f"{split} must not be empty")
        if split == "val" and subset.label.nunique() != 2:
            raise ValueError("val requires both classes for validation threshold selection")


def validate_comparison(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Require identical held-out transaction IDs, splits and labels in both models."""
    validate_predictions(left)
    validate_predictions(right)
    columns = ["edge_id", "split", "label"]
    a = left.loc[left.split.isin(["val", "test"]), columns].sort_values("edge_id")
    b = right.loc[right.split.isin(["val", "test"]), columns].sort_values("edge_id")
    if a.to_dict("records") != b.to_dict("records"):
        raise ValueError("Models must cover identical held-out edge IDs, splits and labels")


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Maximise validation F1; ties choose the highest cutoff deterministically."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(labels) != len(scores) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection needs aligned binary examples of both classes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid threshold scores")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    candidates = np.flatnonzero(np.isclose(f1, f1.max(), rtol=1e-12, atol=1e-12))
    return float(thresholds[candidates[-1]])


def split_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Compute AP from scores and classification metrics at one fixed threshold."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, scores >= threshold, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel()
    return {
        "events": int(len(labels)),
        "positives": int(np.sum(labels)),
        "prevalence": float(np.mean(labels)),
        "average_precision": float(average_precision_score(labels, scores))
        if np.sum(labels)
        else None,
        "positive_class_status": "available" if np.sum(labels) else "no positives; AP undefined",
        "zero_division": 0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_predictions(frame: pd.DataFrame) -> dict:
    """Freeze validation threshold before reading test labels for test metrics."""
    validate_predictions(frame)
    val = frame.loc[frame.split == "val"]
    threshold = select_threshold(val.label.to_numpy(), val.score.to_numpy())
    result = {
        "run_id": str(frame.run_id.iloc[0]),
        "model": str(frame.model.iloc[0]),
        "threshold": threshold,
        "threshold_method": "maximum validation F1; highest threshold on ties",
        "metric_definition": "AP = sklearn average_precision_score, not trapezoidal PR-AUC",
    }
    for split in ["val", "test"]:
        part = frame.loc[frame.split == split]
        result[split] = split_metrics(part.label.to_numpy(), part.score.to_numpy(), threshold)
    return result

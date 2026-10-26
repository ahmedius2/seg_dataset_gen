"""Occupancy metrics computed over observed cells only."""

import numpy as np
import torch


@torch.no_grad()
def confusion_counts(logits, target, weight, threshold=0.5):
    """Return (tp, fp, fn, tn) as ints, ignoring weight == 0 cells."""
    pred = (torch.sigmoid(logits) >= threshold).float()
    valid = weight > 0.5
    pred, tgt = pred[valid], target[valid]

    tp = ((pred == 1) & (tgt == 1)).sum().item()
    fp = ((pred == 1) & (tgt == 0)).sum().item()
    fn = ((pred == 0) & (tgt == 1)).sum().item()
    tn = ((pred == 0) & (tgt == 0)).sum().item()
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn, eps=1e-9):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    return {"precision": precision, "recall": recall, "f1": f1,
            "iou": iou, "accuracy": accuracy}


@torch.no_grad()
def average_precision(logits, target, weight, num_bins=200):
    """
    Area under the precision-recall curve over observed cells.

    Uses a probability histogram rather than a full sort, so it stays cheap
    on large grids.
    """
    prob = torch.sigmoid(logits)[weight > 0.5].detach().cpu().numpy()
    tgt = target[weight > 0.5].detach().cpu().numpy().astype(np.float64)

    if prob.size == 0 or tgt.sum() == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_idx = np.clip(np.digitize(prob, edges) - 1, 0, num_bins - 1)

    pos = np.bincount(bin_idx, weights=tgt, minlength=num_bins)
    cnt = np.bincount(bin_idx, minlength=num_bins)

    tp = np.cumsum(pos[::-1])
    fp = np.cumsum((cnt - pos)[::-1])
    total_pos = pos.sum()

    recall = tp / max(total_pos, 1e-9)
    precision = tp / np.maximum(tp + fp, 1e-9)

    ap = 0.0
    for r in np.linspace(0, 1, 11):
        m = recall >= r
        ap += (precision[m].max() if m.any() else 0.0) / 11.0
    return float(ap)
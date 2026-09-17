"""Loss functions for dense binary occupancy prediction."""

import torch
import torch.nn.functional as F


def focal_loss(logits, target, weight=None, alpha=0.75, gamma=2.0,
               reduction="mean"):
    """Binary focal loss with an optional per-cell weight mask."""
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce

    if weight is not None:
        loss = loss * weight
        if reduction == "mean":
            return loss.sum() / weight.sum().clamp_min(1.0)
        if reduction == "sum":
            return loss.sum()
        return loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def bce_loss(logits, target, weight=None, pos_weight=1.0, reduction="mean"):
    pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pw)
    if weight is not None:
        loss = loss * weight
        if reduction == "mean":
            return loss.sum() / weight.sum().clamp_min(1.0)
        if reduction == "sum":
            return loss.sum()
        return loss
    return loss.mean() if reduction == "mean" else loss.sum()


def soft_iou_loss(logits, target, weight=None, eps=1.0):
    """1 - soft IoU, averaged over the batch."""
    prob = torch.sigmoid(logits)
    if weight is not None:
        prob = prob * weight
        target = target * weight
    dims = tuple(range(1, prob.dim()))
    inter = (prob * target).sum(dim=dims)
    union = prob.sum(dim=dims) + target.sum(dim=dims) - inter
    iou = (inter + eps) / (union + eps)
    return 1.0 - iou.mean()


def build_loss(cfg):
    """Return loss(logits, target, weight) -> scalar tensor."""
    def _loss(logits, target, weight):
        if cfg.loss_type == "focal":
            main = focal_loss(logits, target, weight,
                              alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
        elif cfg.loss_type == "bce":
            main = bce_loss(logits, target, weight, pos_weight=cfg.pos_weight)
        elif cfg.loss_type == "soft_iou":
            main = soft_iou_loss(logits, target, weight)
        else:
            raise ValueError(f"unknown loss_type: {cfg.loss_type}")

        if cfg.dice_weight > 0.0:
            main = main + cfg.dice_weight * soft_iou_loss(logits, target, weight)
        return main

    return _loss
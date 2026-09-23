"""Training entry point. The network itself is supplied by build_model()."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont



from .config import Config
from .dataset import (BEVOccupancyDataset, collate_fn, discover_files,
                      split_files)
from .losses import build_loss
from .metrics import average_precision, confusion_counts, metrics_from_counts

from .model.dynamic_pillars import DynamicPillarVFE
from .model.pillar_scatter import PointPillarScatter
from .model.base_bev_backbone import BaseBEVBackbone
from .model.centerhead import CenterHead

# --------------------------------------------------------------------------- #
#  Model factory -- replace the body with your pillar backbone
# --------------------------------------------------------------------------- #

class PillarBasedModel(torch.nn.Module):
    def __init__(self, vfe, scatter, backbone, head):
        super().__init__()
        self.vfe = vfe
        self.scatter = scatter
        self.backbone = backbone
        self.head = head

    def forward(self, points):
        # points: [M, 4] = (batch_idx, x, y, z) in meters, no intensity
        voxel_coords, pillar_features = self.vfe(points)
        spatial_features = self.scatter(voxel_coords, pillar_features)
        features = self.backbone(spatial_features)
        logits = self.head(features)
        return logits.squeeze(1)

def build_model(cfg: Config):
    """
    Return an nn.Module with signature:

        forward(points: [M, 4]) -> logits [B, H, W]

    `points` is (batch_idx, x, y, z) in meters with no intensity.
    Output must be raw logits, not probabilities.
    """

    vfe = DynamicPillarVFE(model_cfg=cfg.vfe_config, num_point_features=cfg.num_point_features,
                           voxel_size=cfg.pillar_dims, grid_size=cfg.grid_size,
                           point_cloud_range=cfg.point_cloud_range)
    scatter = PointPillarScatter(model_cfg=cfg.scatter_config, grid_size=cfg.grid_size)
    backbone = BaseBEVBackbone(model_cfg=cfg.backbone_2d_config, input_channels=cfg.scatter_config['NUM_BEV_FEATURES'])
    head = CenterHead(input_channels=backbone.num_bev_features)

    return PillarBasedModel(vfe=vfe, scatter=scatter, backbone=backbone, head=head)

# --------------------------------------------------------------------------- #

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: Config):
    files = discover_files(cfg)
    train_f, val_f, test_f = split_files(files, cfg)

    print(f"[data] total={len(files)} train={len(train_f)} "
          f"val={len(val_f)} test={len(test_f)}")

    train_ds = BEVOccupancyDataset(train_f, cfg, train=True)
    val_ds = BEVOccupancyDataset(val_f, cfg, train=False)
    test_ds = BEVOccupancyDataset(test_f, cfg, train=False)

    pin = torch.cuda.is_available()
    common = dict(collate_fn=collate_fn, num_workers=cfg.num_workers,
                  pin_memory=pin)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        drop_last=len(train_ds) > cfg.batch_size,
        persistent_workers=cfg.num_workers > 0, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            persistent_workers=cfg.num_workers > 0, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             **common)
    return train_loader, val_loader, test_loader


def make_optimizer(model, cfg: Config):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.99))


def make_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    if cfg.scheduler == "none":
        return None
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = max(1, cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        if cfg.scheduler == "cosine":
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        if cfg.scheduler == "step":
            return 0.1 ** int(progress * 3)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, device, cfg, optimizer=None,
              scaler=None, epoch=0, log_every=20):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    n_batches = 0
    tp = fp = fn = tn = 0
    ap_sum, ap_n = 0.0, 0

    for step, (pts, tgt, wgt, _meta) in enumerate(loader):
        pts = pts.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        wgt = wgt.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type,
                                enabled=cfg.amp and device.type == "cuda"):
                logits = model(pts)
                loss = criterion(logits.float(), tgt, wgt)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if cfg.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       cfg.grad_clip)
                    optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            b_tp, b_fp, b_fn, b_tn = confusion_counts(
                logits.float(), tgt, wgt, cfg.eval_threshold)
            tp += b_tp; fp += b_fp; fn += b_fn; tn += b_tn
            ap = average_precision(logits.float(), tgt, wgt)
            if not np.isnan(ap):
                ap_sum += ap
                ap_n += 1

        if train and log_every and (step + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch} step {step + 1}/{len(loader)} "
                  f"loss={total_loss / n_batches:.4f} lr={lr:.2e}")

    stats = metrics_from_counts(tp, fp, fn, tn)
    stats["loss"] = total_loss / max(1, n_batches)
    stats["ap"] = ap_sum / ap_n if ap_n else float("nan")
    return stats


@torch.no_grad()
def save_validation_visualizations(model, loader, device, cfg, epoch, out_dir):
    """Save observed mask, target, confidence, and thresholded predictions.

    Each validation sample is written as its own labeled PNG inside a
    per-epoch directory: <out_dir>/val_epoch_<epoch>/.
    """
    model.eval()

    # Dedicated directory for this epoch's generated images.
    epoch_dir = Path(out_dir) / f"val_epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # Try to load a font once; fall back to PIL's default bitmap font.
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    panel_titles = ["Observed", "Target", "Probability", "Prediction"]

    # Iterate over every batch in the loader instead of only the first one.
    for batch_index, batch in enumerate(loader):
        points, target, weight, _ = batch
        logits = model(points.to(device, non_blocking=True)).float().cpu()
        probability = torch.sigmoid(logits)
        prediction = probability >= cfg.eval_threshold

        for index in range(target.shape[0]):
            observed = weight[index].numpy() > 0.5
            target_image = target[index].numpy() > 0.5
            prob_image = probability[index].numpy()
            pred_image = prediction[index].numpy()

            panels = [
                np.where(observed, 255, 32).astype(np.uint8),
                (target_image * 255).astype(np.uint8),
                (prob_image * 255).clip(0, 255).astype(np.uint8),
                (pred_image * 255).astype(np.uint8),
            ]

            # Stack panels horizontally, then add a labeled header strip.
            strip = np.concatenate(panels, axis=1)
            panel_width = strip.shape[1] // len(panels)
            header_height = 24

            canvas = Image.new("L", (strip.shape[1], strip.shape[0] + header_height), 0)
            canvas.paste(Image.fromarray(strip, mode="L"), (0, header_height))

            draw = ImageDraw.Draw(canvas)
            for panel_index, title in enumerate(panel_titles):
                x = panel_index * panel_width + 6
                draw.text((x, 5), title, fill=255, font=font)

            # Global sample index across all batches.
            sample_index = batch_index * target.shape[0] + index
            filename = epoch_dir / f"val_epoch_{epoch:03d}_sample_{sample_index:04d}.png"
            canvas.save(filename)


def format_stats(tag, stats):
    return (f"[{tag}] loss={stats['loss']:.4f} "
            f"iou={stats['iou']:.4f} f1={stats['f1']:.4f} "
            f"prec={stats['precision']:.4f} rec={stats['recall']:.4f} "
            f"ap={stats['ap']:.4f}")


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                    best_iou, cfg):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "best_iou": best_iou,
        "config": cfg.__dict__,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="override data_root")
    parser.add_argument("--out", default=None, help="override out_dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss", default=None,
                        choices=["focal", "bce", "soft_iou"])
    parser.add_argument("--config", default=None,
                        help="JSON config to load instead of defaults")
    parser.add_argument("--resume", default=None, help="checkpoint to resume")
    args = parser.parse_args()

    cfg = Config.from_json(args.config) if args.config else Config()
    if args.data: cfg.data_root = args.data
    if args.out: cfg.out_dir = args.out
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr
    if args.loss: cfg.loss_type = args.loss
    cfg.__post_init__()

    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    # add a number to out_dir, if it already exists, increment the number until we find a free one
    if out_dir.exists():
        i = 1
        while (out_dir.parent / f"{out_dir.name}_{i:03d}").exists():
            i += 1
        out_dir = out_dir.parent / f"{out_dir.name}_{i:03d}"
    out_dir.mkdir(parents=True, exist_ok=False)
    cfg.to_json(out_dir / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_loader, val_loader, _ = build_loaders(cfg)

    sample = next(iter(val_loader))
    valid_positive = int((sample[1] * sample[2]).sum().item())
    print(f"[labels] validation batch valid positives={valid_positive}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params / 1e6:.2f} M trainable parameters")

    criterion = build_loss(cfg)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == "cuda")

    start_epoch = 0
    best_iou = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", -1.0)
        print(f"[resume] from {args.resume} at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, cfg,
                       optimizer=optimizer, scaler=scaler, epoch=epoch,
                       log_every=cfg.log_every)
        if scheduler:
            scheduler.step()
        va = run_epoch(model, val_loader, criterion, device, cfg,
                       optimizer=None, epoch=epoch, log_every=0)

        save_validation_visualizations(model, val_loader, device, cfg, epoch,
                          out_dir)

        print(f"epoch {epoch} ({time.time() - t0:.1f}s)")
        print("  " + format_stats("train", tr))
        print("  " + format_stats("val  ", va))

        history.append({"epoch": epoch, "train": tr, "val": va})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                        scaler, epoch, best_iou, cfg)
        if va["iou"] > best_iou:
            best_iou = va["iou"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler,
                            scaler, epoch, best_iou, cfg)
            print(f"  new best val IoU {best_iou:.4f} -> best.pt")

        if cfg.save_every and (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                            scheduler, scaler, epoch, best_iou, cfg)

    print(f"done. best val IoU = {best_iou:.4f}")


if __name__ == "__main__":
    main()
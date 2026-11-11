"""Evaluate a checkpoint on the val/test split."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import (BEVOccupancyDataset, collate_fn, discover_files,
                      split_files)
from .losses import build_loss
from .metrics import average_precision, confusion_counts, metrics_from_counts
from .train import build_model


class OnnxEvaluationModel:
    def __init__(self, model):
        self.model = model
        self.dropped_frames = 0
        self.dropped_pillars = 0

    def eval(self):
        return self

    def predict_batch(self, points, batch_size):
        points = points.cpu().numpy()
        logits = []
        for index in range(batch_size):
            logits.append(self.model.predict_logits(points[points[:, 0] == index, 1:4]))
            dropped = getattr(self.model, "dropped_pillars", 0)
            self.dropped_frames += int(dropped > 0)
            self.dropped_pillars += dropped
        return torch.from_numpy(np.stack(logits))


def model_logits(model, points, batch_size):
    if isinstance(model, OnnxEvaluationModel):
        return model.predict_batch(points, batch_size)
    return model(points)


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg, threshold=0.5,
             collect_per_frame=False):
    """
    Run the model over a loader and aggregate occupancy metrics.

    Returns a dict with the aggregate metrics plus, optionally, a per-frame
    list of {path, iou, precision, recall, n_points, n_occupied}.
    """
    model.eval()

    tp = fp = fn = tn = 0
    loss_sum, n_batches = 0.0, 0
    ap_sum, ap_n = 0.0, 0
    per_frame = []

    for pts, tgt, wgt, metas in loader:
        pts = pts.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        wgt = wgt.to(device, non_blocking=True)

        logits = model_logits(model, pts, tgt.shape[0]).float()

        loss_sum += criterion(logits, tgt, wgt).item()
        n_batches += 1

        b_tp, b_fp, b_fn, b_tn = confusion_counts(logits, tgt, wgt, threshold)
        tp += b_tp
        fp += b_fp
        fn += b_fn
        tn += b_tn

        ap = average_precision(logits, tgt, wgt)
        if not np.isnan(ap):
            ap_sum += ap
            ap_n += 1

        if collect_per_frame:
            prob = torch.sigmoid(logits)
            for k, meta in enumerate(metas):
                valid = wgt[k] > 0.5
                pred = (prob[k] >= threshold)[valid]
                true = tgt[k][valid] > 0.5

                p_tp = (pred & true).sum().item()
                p_fp = (pred & ~true).sum().item()
                p_fn = (~pred & true).sum().item()

                eps = 1e-9
                prec = p_tp / (p_tp + p_fp + eps)
                rec = p_tp / (p_tp + p_fn + eps)
                per_frame.append({
                    "path": meta["path"],
                    "iou": p_tp / (p_tp + p_fp + p_fn + eps),
                    "precision": prec,
                    "recall": rec,
                    "f1": 2 * prec * rec / (prec + rec + eps),
                    "n_points": meta["n_points"],
                    "n_occupied": meta["n_occupied"],
                })

    stats = metrics_from_counts(tp, fp, fn, tn)
    stats["loss"] = loss_sum / max(1, n_batches)
    stats["ap"] = ap_sum / ap_n if ap_n else float("nan")
    stats["threshold"] = threshold
    stats["n_frames"] = len(loader.dataset)
    stats["tp"], stats["fp"], stats["fn"], stats["tn"] = tp, fp, fn, tn

    if collect_per_frame:
        stats["per_frame"] = per_frame
    return stats


def format_stats(tag, stats):
    return (f"[{tag}] loss={stats['loss']:.4f} "
            f"iou={stats['iou']:.4f} f1={stats['f1']:.4f} "
            f"prec={stats['precision']:.4f} rec={stats['recall']:.4f} "
            f"ap={stats['ap']:.4f} acc={stats['accuracy']:.4f}")


def sweep_thresholds(model, loader, device, cfg, thresholds=None):
    """
    Evaluate at several decision thresholds in a single pass.

    Returns {threshold: metrics_dict}. Useful for picking the operating point
    that matches the precision/recall trade-off you want.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    model.eval()
    counts = {float(t): [0, 0, 0, 0] for t in thresholds}

    with torch.no_grad():
        for pts, tgt, wgt, _ in loader:
            pts = pts.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            wgt = wgt.to(device, non_blocking=True)

            prob = torch.sigmoid(model_logits(model, pts, tgt.shape[0]).float())
            valid = wgt > 0.5
            prob = prob[valid]
            true = tgt[valid] > 0.5

            for t in thresholds:
                pred = prob >= float(t)
                c = counts[float(t)]
                c[0] += (pred & true).sum().item()
                c[1] += (pred & ~true).sum().item()
                c[2] += (~pred & true).sum().item()
                c[3] += (~pred & ~true).sum().item()

    out = {}
    for t, (tp, fp, fn, tn) in counts.items():
        m = metrics_from_counts(tp, fp, fn, tn)
        m["threshold"] = t
        out[t] = m
    return out


def load_model(ckpt_path, device, cfg_override=None):
    """Rebuild the model from a checkpoint and load its weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_cfg = Config.from_dict(ckpt["config"])
    if cfg_override is None:
        cfg = checkpoint_cfg
    else:
        cfg_override.assert_model_compatible(checkpoint_cfg)
        cfg = cfg_override
    cfg.__post_init__()

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="training run directory containing config.yaml and best.pt")
    parser.add_argument("--backend", choices=["torch", "onnx", "tidl"], default="torch")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Torch device")
    parser.add_argument("--pfn-on-cpu", action="store_true", help="use CPU PFN with TIDL")
    parser.add_argument("--voxelizer-threads", type=int)
    parser.add_argument("--data", default=None, help="override data_root")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="override data loader workers")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sweep", action="store_true",
                        help="also sweep decision thresholds")
    parser.add_argument("--per-frame", action="store_true",
                        help="include per-frame metrics in the JSON output")
    parser.add_argument("--out", default=None,
                        help="JSON file to write results to")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "best.pt"
    if args.backend == "torch":
        if args.pfn_on_cpu:
            parser.error("--pfn-on-cpu requires --backend tidl")
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA requested but unavailable")
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                              if args.device == "auto" else args.device)
        model, cfg, ckpt = load_model(
            ckpt_path, device, Config.from_yaml(run_dir / "config.yaml"))
        if args.voxelizer_threads is not None:
            if args.voxelizer_threads < 1:
                parser.error("--voxelizer-threads must be positive")
            if cfg.vfe_type != "static" or cfg.model_variant == "heightmap":
                parser.error("--voxelizer-threads requires a static pillar model")
            model.vfe.num_threads = args.voxelizer_threads
    else:
        if args.device != "auto":
            parser.error("--device applies only to the torch backend")
        if args.pfn_on_cpu and args.backend != "tidl":
            parser.error("--pfn-on-cpu requires --backend tidl")
        if args.voxelizer_threads is not None and args.voxelizer_threads < 1:
            parser.error("--voxelizer-threads must be positive")
        from .inference import load_run_model
        options = SimpleNamespace(run_dir=run_dir, backend=args.backend, device="auto",
                                  pfn_on_cpu=args.pfn_on_cpu,
                                  voxelizer_threads=args.voxelizer_threads)
        backend_model, cfg, _ = load_run_model(options)
        model, ckpt = OnnxEvaluationModel(backend_model), None
        device = torch.device("cpu")
    print(f"[device] {device} ({args.backend})")
    if args.data:
        cfg.data_root = args.data
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.workers is not None:
        if args.workers < 0:
            parser.error("--workers must be nonnegative")
        cfg.num_workers = args.workers
    threshold = args.threshold if args.threshold is not None \
        else cfg.eval_threshold

    if ckpt is not None:
        print(f"[ckpt] {ckpt_path} (epoch {ckpt.get('epoch')}, "
              f"best val IoU {ckpt.get('best_iou', float('nan')):.4f})")

    files = discover_files(cfg)
    train_f, val_f, test_f = split_files(files, cfg)
    split_map = {"train": train_f, "val": val_f, "test": test_f}
    split_files_list = split_map[args.split]
    if not split_files_list:
        raise SystemExit(f"split '{args.split}' is empty")

    ds = BEVOccupancyDataset(split_files_list, cfg, train=False)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda")

    criterion = build_loss(cfg)
    stats = evaluate(model, loader, criterion, device, cfg,
                     threshold=threshold,
                     collect_per_frame=args.per_frame)

    print(f"[split] {args.split} ({len(ds)} frames)")
    print(format_stats(args.split, stats))

    result = {"split": args.split, "backend": args.backend,
              "run_dir": str(run_dir), "metrics": stats}
    if isinstance(model, OnnxEvaluationModel):
        result["dropped_pillar_frames"] = model.dropped_frames
        result["dropped_pillars"] = model.dropped_pillars
        if model.dropped_pillars:
            print(f"[warning] {model.dropped_pillars} pillars dropped across "
                  f"{model.dropped_frames} frames by the export limit")

    if args.sweep:
        sweep = sweep_thresholds(model, loader, device, cfg)
        result["threshold_sweep"] = {str(k): v for k, v in sweep.items()}
        print("\nthreshold sweep:")
        print("  thr    iou     f1      prec    rec")
        for t in sorted(sweep):
            m = sweep[t]
            print(f"  {t:.2f}  {m['iou']:.4f}  {m['f1']:.4f}  "
                  f"{m['precision']:.4f}  {m['recall']:.4f}")

    if args.per_frame:
        frames = stats["per_frame"]
        worst = sorted(frames, key=lambda f: f["iou"])[:10]
        print("\nworst 10 frames by IoU:")
        for f in worst:
            print(f"  iou={f['iou']:.4f} prec={f['precision']:.4f} "
                  f"rec={f['recall']:.4f} occ={f['n_occupied']:6d} "
                  f"{Path(f['path']).name}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
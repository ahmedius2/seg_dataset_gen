"""SemanticKITTI point-wise ground-segmentation evaluation.

Reports the protocol used by the KAIST ground-segmentation benchmark
(Patchwork / Patchwork++ / TRAVEL), so the numbers can be compared directly
with that literature:

* every point receives the prediction of its BEV cell: ground when
  ``p(occupied) < threshold``, non-ground otherwise;
* ground labels are ``road, parking, sidewalk, other-ground, lane-marking,
  terrain``; ``unlabeled``/``outlier`` are never scored;
* vegetation is excluded by default (the benchmark's "w/o vegetation"
  column). ``--with-vegetation`` reproduces its "w/ vegetation" column, where
  vegetation below ``-(3/4) * 1.723 m`` also counts as ground;
* Precision, Recall and F1 are averaged per frame, and the reported F1 is the
  harmonic mean of the averaged P and R, as in the published tables.

Two scopes are available because a BEV model only sees a finite window:

* ``--scope window`` (default): score points inside the model's BEV window.
  This is the model-quality number comparable with the per-scan methods.
* ``--scope scan``: score every labeled point; points outside the window are
  predicted non-ground, i.e. what the sensor-facing system would output.

Usage (from ``lidar/``)::

    # evaluate a run directory as it was configured (e.g. trained on KITTI)
    python -m gnd_seg_dnn.evaluate_kitti --run-dir kitti_statpillars_dseg_04 --split test

    # evaluate an existing simulator-trained checkpoint on SemanticKITTI
    python -m gnd_seg_dnn.evaluate_kitti --run-dir statpillars_xyz_stats_04 \
        --data /mnt/data/SemanticKITTI/raw/dataset/sequences \
        --sequences 08 --scope window
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .dataset import cell_indices
from .kitti import (KITTI_GROUND_CLASSES, KITTI_VEGETATION_CLASSES, load_frame,
                    normalize_sequences)

# HDL-64E mounting height assumed by the benchmark; vegetation below
# -(3/4) of it counts as ground in the "with vegetation" variant.
SENSOR_HEIGHT_M = 1.723
VEGETATION_THR_M = -SENSOR_HEIGHT_M * 3.0 / 4.0

# GndNet's ground definition: road, parking, sidewalk, other-ground,
# lane-marking, terrain. Vegetation is NOT ground there (unlike its README).
GNDNET_GROUND_CLASSES = (40, 44, 48, 49, 60, 72)

# Its published 7/4 sequence split (SemanticKITTI_GndNet_data.zip).
GNDNET_TRAIN_SEQUENCES = ["00", "02", "03", "04", "06", "08", "10"]
GNDNET_VAL_SEQUENCES = ["01", "05", "07", "09"]


def load_model(run_dir, device_name="auto", backend="torch"):
    """Load config + model from a run directory, honouring its own geometry."""
    run_dir = Path(run_dir)
    cfg = Config.from_yaml(run_dir / "config.yaml")

    if backend == "torch":
        device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device_name == "auto" else device_name)
        from .train import build_model
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu",
                          weights_only=False)
        checkpoint_cfg = Config.from_dict(ckpt["config"])
        # The checkpoint is authoritative for anything that shapes the model;
        # only data-loading fields may differ.
        cfg.assert_model_compatible(checkpoint_cfg)
        cfg = checkpoint_cfg
        model = build_model(cfg)
        model.load_state_dict(ckpt["model"], strict=True)
        return model.to(device).eval(), cfg, device, ckpt.get("epoch")

    from .inference import load_run_model
    options = type("Options", (), {
        "run_dir": run_dir, "backend": backend, "device": device_name,
        "pfn_on_cpu": False, "voxelizer_threads": None})()
    wrapped, export_cfg, device_name = load_run_model(options)
    return wrapped, export_cfg, torch.device("cpu"), None


class BackendAdapter:
    """Uniform ``predict(points [N,3]) -> prob [H, W]`` over torch / ONNX."""

    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.is_onnx = hasattr(model, "predict_logits")

    @torch.no_grad()
    def predict(self, xyz):
        points = torch.from_numpy(
            np.concatenate([np.zeros((xyz.shape[0], 1), np.float32),
                            xyz.astype(np.float32)], axis=1))
        if self.is_onnx:
            logits = self.model.predict_logits(xyz.astype(np.float32))
            return torch.sigmoid(torch.from_numpy(logits)).numpy()
        if self.cfg.model_variant == "statpillars":
            logits = self.model(points, batch_size=1)
        else:
            logits = self.model(points.to(self.device))
        return torch.sigmoid(logits.float())[0].cpu().numpy()


def frame_counts(prob, xyz, classes, cfg, threshold, scope="window",
                 with_vegetation=False, protocol="kaist"):
    """Return ``(tp, fp, fn, n_points, n_in_grid, n_scored)`` for one frame.

    ``protocol="kaist"`` is the KAIST ground-segmentation benchmark: ground
    when ``p(occupied) < threshold``, ground labels from
    ``KITTI_GROUND_CLASSES``, vegetation not scored.

    ``protocol="gndnet"`` reproduces the GndNet evaluation
    (``evaluate_SemanticKITTI.py``) exactly: points outside the BEV window and
    unlabeled/outlier points are dropped, only the first four ground classes
    (no vegetation) count as ground, and the decision comes from the same
    occupancy probability threshold.
    """
    i, j, keep = cell_indices(xyz, cfg)
    in_grid = int(keep.sum())
    n_points = int(xyz.shape[0])

    if protocol == "gndnet":
        ground = np.isin(classes, GNDNET_GROUND_CLASSES)
        scored = np.zeros(n_points, dtype=bool)
        scored[keep] = True
        scored &= ~np.isin(classes, (0, 1))
        pred_ground = np.zeros(n_points, dtype=bool)
        pred_ground[keep] = prob[i, j] < threshold
    else:
        ground = np.isin(classes, KITTI_GROUND_CLASSES)
        if with_vegetation:
            ground = ground | (np.isin(classes, KITTI_VEGETATION_CLASSES)
                               & (xyz[:, 2] < VEGETATION_THR_M))
            scored = ~np.isin(classes, (0, 1))
        else:
            scored = ~np.isin(classes, (0, 1, 70))
        if scope == "scan":
            # Out-of-window points cannot be observed by the model: predict
            # non-ground for them (deployment behaviour).
            pred_ground = np.zeros(n_points, dtype=bool)
            pred_ground[keep] = prob[i, j] < threshold
        else:
            pred_ground = prob[i, j] < threshold
            scored = scored[keep]
            ground = ground[keep]

    pred, gt = pred_ground[scored], ground[scored]
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    return tp, fp, fn, n_points, in_grid, int(scored.sum())


def _prf(tp, fp, fn, eps=1e-9):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return precision, recall


def _harmonic(p, r, eps=1e-9):
    return 2 * p * r / (p + r + eps)


def _summary(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def height_threshold_predict(xyz, cfg, reach_m=0.2):
    """Classical ground segmentation: distance above the local cell floor.

    For each BEV cell, take the lowest point as the local ground level, then
    call a point ground when it sits at most ``reach_m`` above that floor.
    This is roughly GndNet's own decision rule (elevation + threshold), with the
    learned elevation surface replaced by the cell minimum, so it isolates how
    much of the task is geometry rather than learning.

    Returns ``prob`` [H, W] shaped like the model output: cells holding any
    point above the floor get high occupancy probability, so the same
    ``prob < threshold`` rule in ``frame_counts`` applies unchanged.
    """
    h, w = cfg.grid_h, cfg.grid_w
    prob = np.ones((h, w), dtype=np.float32)      # empty cells read as ground
    i, j, keep = cell_indices(xyz, cfg)
    if keep.sum() == 0:
        return prob

    z = xyz[keep, 2]
    rows, cols = i, j
    # Per-cell floor via a dense scatter-min.
    floor = np.full((h, w), np.inf, dtype=np.float32)
    np.minimum.at(floor, (rows, cols), z)
    above = z - floor[rows, cols]

    # Cells whose points all sit at the floor are ground (prob low).
    cell_max_above = np.zeros((h, w), dtype=np.float32)
    np.maximum.at(cell_max_above, (rows, cols), above)
    visited = np.zeros((h, w), dtype=bool)
    visited[rows, cols] = True

    # A cell reads as occupied when it contains a point higher than reach_m.
    prob[visited] = np.where(cell_max_above[visited] > reach_m, 1.0, 0.0)
    return prob


def reference_baselines(cfg, files, threshold, scope="window",
                        with_vegetation=False, protocol="kaist",
                        height_reach=(0.1, 0.2, 0.3)):
    """Oracle / all-ground / all-obstacle references on the same frames.

    Point-wise ground metrics saturate easily: predicting "everything is
    ground" already scores highly, and the cell-quantisation ceiling is below
    100. These references put a result in context.
    """
    from .dataset import rasterize_masks
    from .kitti import KITTI_GROUND_CLASSES

    sums = {name: np.zeros(4, dtype=np.int64)
            for name in ("oracle", "all_ground", "all_obstacle")}
    for reach in height_reach:
        sums[f"height_{reach:g}m"] = np.zeros(4, dtype=np.int64)
    zeros = np.zeros((cfg.grid_h, cfg.grid_w), dtype=np.float32)

    for path in files:
        xyz, classes = load_frame(path)
        finite = np.isfinite(xyz).all(axis=1)
        if not finite.all():
            xyz, classes = xyz[finite], classes[finite]

        ground_classes = (GNDNET_GROUND_CLASSES if protocol == "gndnet"
                          else KITTI_GROUND_CLASSES)
        # The scored mask must match frame_counts(), otherwise the oracle is
        # built from a different point set than it is scored on. GndNet scores
        # vegetation as non-ground, so only unlabeled/outlier are dropped.
        if with_vegetation:
            scored = ~np.isin(classes, (0, 1))
        elif protocol == "gndnet":
            scored = ~np.isin(classes, (0, 1))
        else:
            scored = ~np.isin(classes, (0, 1, 70))
        positive = scored & ~np.isin(classes, ground_classes)
        oracle, _ = rasterize_masks(xyz, cfg, positive, scored)
        ones = np.ones_like(zeros)

        predictors = [("oracle", oracle), ("all_ground", zeros),
                      ("all_obstacle", ones)]
        predictors += [(f"height_{r:g}m", height_threshold_predict(xyz, cfg, r))
                       for r in height_reach]

        for name, prob in predictors:
            tp, fp, fn, _, _, n_scored = frame_counts(
                prob, xyz, classes, cfg, threshold, scope, with_vegetation,
                protocol=protocol)
            tn = n_scored - tp - fp - fn
            sums[name] += np.array([tp, fp, fn, tn], dtype=np.int64)

    out = {}
    for name, (tp, fp, fn, tn) in sums.items():
        p, r = _prf(int(tp), int(fp), int(fn))
        iou_g = tp / max(1, tp + fp + fn)
        iou_n = tn / max(1, tn + fp + fn)
        out[name] = {"precision": p, "recall": r, "f1": _harmonic(p, r),
                     "miou": float(0.5 * (iou_g + iou_n)),
                     "tp": int(tp), "fp": int(fp), "fn": int(fn)}
    return out


def evaluate_frames(adapter, cfg, files, threshold, scope="window",
                    with_vegetation=False, verbose=True, protocol="kaist"):
    frames = []
    for index, path in enumerate(files):
        xyz, classes = load_frame(path)
        finite = np.isfinite(xyz).all(axis=1)
        if not finite.all():
            xyz, classes = xyz[finite], classes[finite]

        prob = adapter.predict(xyz)
        tp, fp, fn, n_points, n_in_grid, n_scored = frame_counts(
            prob, xyz, classes, cfg, threshold, scope, with_vegetation,
            protocol=protocol)
        p, r = _prf(tp, fp, fn)
        tn = n_scored - tp - fp - fn
        # GndNet's published "mIoU" turns out to be exactly the ground-class
        # IoU (0.841/0.993 -> 0.836), so that is the number to compare against.
        # The two-class mean is also recorded for completeness.
        iou_ground = tp / max(1, tp + fp + fn)
        iou_nonground = tn / max(1, tn + fp + fn)
        frames.append({
            "path": str(path), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": _harmonic(p, r),
            "iou": iou_ground,
            "miou": 0.5 * (iou_ground + iou_nonground),
            "n_points": n_points, "n_in_grid": n_in_grid, "n_scored": n_scored,
        })
        if verbose and (index + 1) % 500 == 0:
            print(f"    {index + 1}/{len(files)} frames", flush=True)

    if not frames:
        return None

    tp = sum(f["tp"] for f in frames)
    fp = sum(f["fp"] for f in frames)
    fn = sum(f["fn"] for f in frames)
    micro_p, micro_r = _prf(tp, fp, fn)
    n_points = sum(f["n_points"] for f in frames)
    n_in_grid = sum(f["n_in_grid"] for f in frames)

    return {
        "n_frames": len(frames),
        "n_points": n_points,
        "n_in_grid": n_in_grid,
        "n_scored": int(sum(f["n_scored"] for f in frames)),
        "coverage": float(n_in_grid / max(1, n_points)),
        "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": _harmonic(micro_p, micro_r),
        "precision": _summary([f["precision"] for f in frames]),
        "recall": _summary([f["recall"] for f in frames]),
        "f1": _summary([f["f1"] for f in frames]),
        "iou": _summary([f["iou"] for f in frames]),
        "miou": _summary([f["miou"] for f in frames]),
        "frames": frames,
    }


def format_table(results):
    head = (f"{'sequence':>9} {'frames':>7} {'cover':>6} "
            f"{'P (%)':>15} {'R (%)':>15} {'F1 (%)':>9} {'gIoU (%)':>9} "
            f"{'mIoU (%)':>9}")
    lines = [head, "-" * len(head),
             "  point-level metrics over scored points: gIoU = ground class, "
             "mIoU = 2-class mean",
             "  binary ground/non-ground task (GndNet/KAIST protocol), NOT the "
             "official SemanticKITTI 19-class mIoU"]
    precs, recs, ious, mious = [], [], [], []
    for name in sorted(results):
        res = results[name]
        if not res:
            continue
        p, r = res["precision"]["mean"] * 100, res["recall"]["mean"] * 100
        f1 = _harmonic(p, r)
        precs.append(p)
        recs.append(r)
        ious.append(res["iou"]["mean"] * 100)
        mious.append(res["miou"]["mean"] * 100)
        lines.append(
            f"{name:>9} {res['n_frames']:>7} {res['coverage'] * 100:>5.1f}% "
            f"{p:>7.2f}±{res['precision']['std'] * 100:<5.2f} "
            f"{r:>7.2f}±{res['recall']['std'] * 100:<5.2f} "
            f"{f1:>8.2f} {res['iou']['mean'] * 100:>9.2f} "
            f"{res['miou']['mean'] * 100:>9.2f}")
    if precs:
        pm, rm = float(np.mean(precs)), float(np.mean(recs))
        lines.append("-" * len(head))
        lines.append(f"{'mean':>9} {'':>7} {'':>6} "
                     f"{pm:>13.2f} {rm:>15.2f} {_harmonic(pm, rm):>9.2f} "
                     f"{float(np.mean(ious)):>9.2f} {float(np.mean(mious)):>9.2f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="SemanticKITTI point-wise ground-segmentation evaluation")
    parser.add_argument("--run-dir", required=True,
                        help="training run directory (config.yaml + best.pt)")
    parser.add_argument("--split", default="test",
                        choices=["val", "test", "train", "all"])
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="override sequences, e.g. --sequences 08 09")
    parser.add_argument("--protocol", choices=["kaist", "gndnet"],
                        default="kaist",
                        help="kaist: KAIST ground-segmentation benchmark "
                             "(default); gndnet: reproduce GndNet's protocol, "
                             "dropping out-of-window and unlabeled points")
    parser.add_argument("--scope", choices=["window", "scan"], default="window",
                        help="window: score in-window points only (default); "
                             "scan: score every labeled point")
    parser.add_argument("--threshold", type=float, default=None,
                        help="occupancy probability threshold (default: config)")
    parser.add_argument("--backend", choices=["torch", "onnx", "tidl"],
                        default="torch")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        default="auto")
    parser.add_argument("--data", default=None,
                        help="override data_root (the 'sequences' directory)")
    parser.add_argument("--train-sequences", nargs="*", default=None)
    parser.add_argument("--val-sequences", nargs="*", default=None)
    parser.add_argument("--test-sequences", nargs="*", default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="limit frames per sequence (0 = all)")
    parser.add_argument("--with-vegetation", action="store_true",
                        help="also score the benchmark's 'w/ vegetation' variant")
    parser.add_argument("--height-baselines", nargs="*", type=float,
                        default=[0.1, 0.2, 0.3],
                        help="reaches (m) for the classical height-threshold "
                             "baseline; pass no values to disable")
    parser.add_argument("--sweep", action="store_true",
                        help="report a threshold sweep on the pooled counts")
    parser.add_argument("--per-frame-out", default=None,
                        help="CSV path for per-frame TP/FP/FN")
    parser.add_argument("--out", default=None, help="JSON output path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    model, cfg, device, epoch = load_model(run_dir, args.device, args.backend)
    adapter = BackendAdapter(model, cfg, device)

    if cfg.dataset != "kitti":
        print(f"[note] run config dataset={cfg.dataset!r}; treating the data as "
              f"SemanticKITTI and keeping the model's trained geometry "
              f"{cfg.grid_w}x{cfg.grid_h} @ {cfg.cell_size} m "
              f"(x {cfg.x_min}..{cfg.x_max}, y {cfg.y_min}..{cfg.y_max})")
    if args.data:
        cfg.data_root = args.data
    for name in ("train_sequences", "val_sequences", "test_sequences"):
        value = getattr(args, f"{name}")
        if value is not None:
            setattr(cfg, name, normalize_sequences(value))
    # A checkpoint trained on another dataset carries no KITTI splits; in that
    # case --sequences selects the frames to evaluate.
    if args.sequences and not (cfg.train_sequences or cfg.val_sequences
                               or cfg.test_sequences):
        cfg.test_sequences = normalize_sequences(args.sequences)
    # --sequences may name sequences outside the configured splits (e.g. the
    # unused 09/10 holdout); discover them into the tested split.
    if args.sequences:
        extra = normalize_sequences(args.sequences)
        known = set(cfg.train_sequences) | set(cfg.val_sequences) | set(cfg.test_sequences)
        unknown = [s for s in extra if s not in known]
        if unknown:
            cfg.test_sequences = list(cfg.test_sequences) + unknown
    if args.frame_stride is not None:
        cfg.frame_stride = args.frame_stride
    cfg.dataset = "kitti"
    if not (cfg.train_sequences or cfg.val_sequences or cfg.test_sequences):
        parser.error("no sequences to evaluate: pass --sequences (e.g. "
                     "--sequences 08) or use a run trained on KITTI")
    cfg.__post_init__()

    threshold = args.threshold if args.threshold is not None else cfg.eval_threshold

    from .dataset import discover_files, split_files
    files = discover_files(cfg)
    train_f, val_f, test_f = split_files(files, cfg)
    selected = {"train": train_f, "val": val_f, "test": test_f,
                "all": list(files)}[args.split]
    if args.sequences:
        wanted = set(normalize_sequences(args.sequences))
        from .kitti import sequence_of
        selected = [f for f in selected if sequence_of(f) in wanted]
    if not selected:
        raise SystemExit(f"no frames selected for split {args.split!r}")

    by_sequence = {}
    from .kitti import sequence_of
    for f in selected:
        by_sequence.setdefault(sequence_of(f), []).append(f)

    print(f"[eval] {args.backend} on {device} | epoch {epoch} | grid "
          f"{cfg.grid_w}x{cfg.grid_h} @ {cfg.cell_size} m | "
          f"protocol {args.protocol} | scope {args.scope} | "
          f"threshold {threshold} | "
          f"{'w/' if args.with_vegetation else 'w/o'} vegetation")

    results, totals = {}, {"tp": 0, "fp": 0, "fn": 0}
    for name in sorted(by_sequence):
        frames = sorted(by_sequence[name])
        if args.max_frames:
            frames = frames[:args.max_frames]
        print(f"  sequence {name}: {len(frames)} frames", flush=True)
        res = evaluate_frames(adapter, cfg, frames, threshold, args.scope,
                              args.with_vegetation, protocol=args.protocol)
        results[name] = res
        if res:
            for key in totals:
                totals[key] += res[key]

    print()
    print(format_table(results))
    mic_p, mic_r = _prf(totals["tp"], totals["fp"], totals["fn"])
    n_scored = sum(res["n_scored"] for res in results.values() if res)
    mic_tn = n_scored - totals["tp"] - totals["fp"] - totals["fn"]
    mic_iou = totals["tp"] / max(1, totals["tp"] + totals["fp"] + totals["fn"])
    mic_nong = mic_tn / max(1, mic_tn + totals["fp"] + totals["fn"])
    print(f"\nmicro (pooled points, point-level): P={mic_p * 100:.2f} "
          f"R={mic_r * 100:.2f} F1={_harmonic(mic_p, mic_r) * 100:.2f} "
          f"gIoU={mic_iou * 100:.2f} mIoU={0.5 * (mic_iou + mic_nong) * 100:.2f}")

    evaluated = [f for name in sorted(by_sequence)
                 for f in sorted(by_sequence[name])[:args.max_frames or None]]
    print("\nreference baselines on the same frames (pooled points):")
    base = reference_baselines(cfg, evaluated, threshold, args.scope,
                               args.with_vegetation, protocol=args.protocol,
                               height_reach=tuple(args.height_baselines))
    print(f"  {'predictor':<22} {'P (%)':>8} {'R (%)':>8} {'F1 (%)':>8} "
          f"{'mIoU (%)':>9}")
    rows = [("all_ground", "all ground"),
            ("all_obstacle", "all obstacle")]
    rows += [(f"height_{r:g}m", f"height rule {r:g} m")
             for r in args.height_baselines
             if f"height_{r:g}m" in base]
    rows += [("oracle", "cell oracle")]
    for name, label in rows:
        b = base[name]
        print(f"  {label:<22} {b['precision'] * 100:>8.2f} "
              f"{b['recall'] * 100:>8.2f} {b['f1'] * 100:>8.2f} "
              f"{b['miou'] * 100:>9.2f}")
    print("  (cell oracle = perfect per-cell decision given this BEV grid, "
          "i.e. the quantisation ceiling)")
    print("  (height rule = classical: point is ground when within REACH of its "
          "cell's lowest point)")

    if args.sweep:
        print("\nthreshold sweep (mean P/R/F1 over sequences):")
        print("   thr    P (%)   R (%)  F1 (%)")
        for t in np.arange(0.05, 1.0, 0.05):
            ps, rs = [], []
            for name in sorted(by_sequence):
                frames = sorted(by_sequence[name])
                if args.max_frames:
                    frames = frames[:args.max_frames]
                res = evaluate_frames(adapter, cfg, frames, float(t), args.scope,
                                      args.with_vegetation, verbose=False,
                                      protocol=args.protocol)
                if res:
                    ps.append(res["precision"]["mean"] * 100)
                    rs.append(res["recall"]["mean"] * 100)
            if ps:
                pm, rm = float(np.mean(ps)), float(np.mean(rs))
                print(f"  {t:4.2f}  {pm:7.2f} {rm:7.2f} {_harmonic(pm, rm):7.2f}")

    if args.per_frame_out:
        path = Path(args.per_frame_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            fh.write("sequence,frame,precision,recall,f1,tp,fp,fn,n_scored\n")
            for name in sorted(results):
                if not results[name]:
                    continue
                for k, f in enumerate(results[name]["frames"]):
                    fh.write(f"{name},{k},{f['precision']:.6f},"
                             f"{f['recall']:.6f},{f['f1']:.6f},"
                             f"{f['tp']},{f['fp']},{f['fn']},{f['n_scored']}\n")
        print(f"wrote {path}")

    if args.out:
        payload = {
            "split": args.split, "scope": args.scope,
            "protocol": args.protocol, "threshold": threshold,
            "with_vegetation": args.with_vegetation, "backend": args.backend,
            "run_dir": str(run_dir), "epoch": epoch,
            "grid": [cfg.grid_w, cfg.grid_h], "cell_size": cfg.cell_size,
            "micro": {"precision": mic_p, "recall": mic_r,
                      "f1": _harmonic(mic_p, mic_r), **totals},
            "baselines": base,
            "sequences": {k: {kk: vv for kk, vv in v.items() if kk != "frames"}
                          for k, v in results.items() if v},
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

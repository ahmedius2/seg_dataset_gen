"""PLY loading, ground-truth rasterization, and the torch Dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config
from .bev_ops import coverage_kernel_size, dilate_bev_mask
from .ply_io import load_cloud, read_ply


# --------------------------------------------------------------------------- #
#  Ground truth
# --------------------------------------------------------------------------- #

def rasterize_occupancy(xyz, intensity, cfg: Config):
    """
    Build the BEV ground-truth mask.

    Returns
    -------
    occ      : [H, W] float32, 1.0 where a point with intensity in
               [lo, hi] landed, else 0.0
    observed : [H, W] bool, True where any point landed
    """
    h, w = cfg.grid_h, cfg.grid_w
    occ = np.zeros((h, w), dtype=np.float32)
    observed = np.zeros((h, w), dtype=bool)

    if xyz.shape[0] == 0:
        return occ, observed

    u = xyz[:, cfg.col_axis]      # -> column index j
    v = xyz[:, cfg.row_axis]      # -> row index i
    z = xyz[:, 2]

    keep = (z >= cfg.z_min) & (z <= cfg.z_max)
    if not keep.any():
        return occ, observed
    u, v, intensity = u[keep], v[keep], intensity[keep]

    j = np.floor((u - cfg.x_min) / cfg.cell_size).astype(np.int64)
    i = np.floor((v - cfg.y_min) / cfg.cell_size).astype(np.int64)

    inside = (j >= 0) & (j < w) & (i >= 0) & (i < h)
    if not inside.any():
        return occ, observed
    i, j, intensity = i[inside], j[inside], intensity[inside]

    if cfg.flip_cols:
        j = (w - 1) - j
    if cfg.flip_rows:
        i = (h - 1) - i

    observed[i, j] = True

    is_occ = (intensity >= cfg.intensity_lo) & (intensity <= cfg.intensity_hi)
    if is_occ.any():
        occ[i[is_occ], j[is_occ]] = 1.0

    # Fill small holes using a fixed metric reach across BEV resolutions.
    occ = dilate_bev_mask(occ, coverage_kernel_size(cfg.cell_size))

    return occ, observed


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #

def discover_files(cfg: Config):
    root = Path(cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    files = sorted(root.glob(cfg.file_glob))
    if not files:
        raise FileNotFoundError(f"no files matched {cfg.file_glob} under {root}")
    return files


def scene_key(path, cfg: Config):
    if not cfg.split_by_scene:
        return str(path)
    rel = Path(path).relative_to(cfg.data_root)
    return rel.parts[0] if len(rel.parts) > 1 else "root"


def split_files(files, cfg: Config):
    """Split into train/val/test, optionally grouped by scene directory."""
    rng = np.random.default_rng(cfg.split_seed)

    if cfg.split_by_scene:
        groups = {}
        for f in files:
            groups.setdefault(scene_key(f, cfg), []).append(f)
        keys = sorted(groups)
        rng.shuffle(keys)

        n = len(keys)
        n_val = max(1, int(round(n * cfg.val_fraction))) if n > 2 else 0
        n_test = int(round(n * cfg.test_fraction)) if n > 4 else 0
        val_keys = set(keys[:n_val])
        test_keys = set(keys[n_val:n_val + n_test])

        train = [f for k in keys if k not in val_keys and k not in test_keys
                 for f in groups[k]]
        val = [f for k in keys if k in val_keys for f in groups[k]]
        test = [f for k in keys if k in test_keys for f in groups[k]]
    else:
        idx = rng.permutation(len(files))
        n_val = int(round(len(files) * cfg.val_fraction))
        n_test = int(round(len(files) * cfg.test_fraction))
        val_idx = set(idx[:n_val].tolist())
        test_idx = set(idx[n_val:n_val + n_test].tolist())
        train = [f for k, f in enumerate(files)
                 if k not in val_idx and k not in test_idx]
        val = [f for k, f in enumerate(files) if k in val_idx]
        test = [f for k, f in enumerate(files) if k in test_idx]

    return train, val, test


class BEVOccupancyDataset(Dataset):
    """
    Yields per frame:

        points : float32 [N, 3]   normalized xyz, no intensity
        target : float32 [H, W]   occupancy target in {0, 1}
        weight : float32 [H, W]   1 where observed, 0 where ignored
        meta   : dict             path, n_points, n_occupied
    """

    def __init__(self, files, cfg: Config, train: bool = False):
        self.files = list(files)
        self.cfg = cfg
        self.train = train
        # Lazy cache: each sample is decoded only when first requested.
        self._cache: list[Any] | None = (
            [None] * len(self.files) if cfg.cache_in_memory else None)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        cfg = self.cfg
        path = self.files[index]

        if self._cache is None:
            xyz, intensity, occ, observed = self._load_sample(path)
        else:
            cached = self._cache[index]
            if cached is None:
                xyz, _intensity, occ, observed = self._load_sample(path)
                cached = (xyz, occ, observed)
                self._cache[index] = cached
            xyz, occ, observed = cached

        xyz = self._augment(xyz, index)

        n = xyz.shape[0]
        max_points = getattr(cfg, "max_points", None)
        if max_points is not None and n > max_points:
            rng = np.random.default_rng(
                (torch.initial_seed() + index) % (2 ** 31) if self.train else index)
            sel = rng.choice(n, size=max_points, replace=False)
            xyz = xyz[sel]

        points = self._normalize(xyz)

        weight = (observed.astype(np.float32) if cfg.ignore_unobserved
                  else np.ones_like(occ, dtype=np.float32))

        meta = {
            "path": str(path),
            "n_points": int(points.shape[0]),
            "n_occupied": int(occ.sum()),
        }
        return points, occ, weight, meta

    def _load_sample(self, path):
        xyz, intensity = load_cloud(path)

        finite = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
        if not finite.all():
            xyz, intensity = xyz[finite], intensity[finite]

        # Build labels from the full cloud before any point subsampling.
        occ, observed = rasterize_occupancy(xyz, intensity, self.cfg)
        return xyz, intensity, occ, observed

    def _augment(self, xyz, index):
        """Z-only geometric augs on a copy of xyz. Labels are not modified."""
        if not self.train:
            return xyz

        cfg = self.cfg
        do_shift = cfg.aug_z_shift
        do_invert = cfg.aug_z_invert and cfg.aug_z_invert_prob > 0
        if not do_shift and not do_invert:
            return xyz

        xyz = xyz.copy()
        rng = np.random.default_rng(
            (torch.initial_seed() + index + 1) % (2 ** 31))

        if do_shift:
            shift = rng.uniform(cfg.aug_z_shift_min, cfg.aug_z_shift_max)
            xyz[:, 2] += np.float32(shift)

        if do_invert and rng.random() < cfg.aug_z_invert_prob:
            xyz[:, 2] *= -1.0

        return xyz

    def _normalize(self, xyz):
        """Return xyz in meters, matching the VFE point_cloud_range."""
        return xyz.astype(np.float32, copy=True)


def collate_fn(batch):
    """Flatten variable-length point sets into [M, 4] = (batch_idx, x, y, z)."""
    points, targets, weights, metas = zip(*batch)

    flat_points = torch.cat([
        torch.cat([torch.full((p.shape[0], 1), k, dtype=torch.float32),
                   torch.from_numpy(p)], dim=1)
        for k, p in enumerate(points)
    ], dim=0)

    tgt = torch.from_numpy(np.stack(targets, axis=0)).float()
    wgt = torch.from_numpy(np.stack(weights, axis=0)).float()
    return flat_points, tgt, wgt, list(metas)
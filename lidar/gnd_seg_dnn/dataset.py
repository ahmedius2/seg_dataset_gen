"""PLY loading, ground-truth rasterization, and the torch Dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config
from .bev_ops import coverage_kernel_size, dilate_bev_mask
from .ply_io import load_cloud, read_ply


def _kitti_ground(classes):
    from .kitti import ground_mask
    return ground_mask(classes)


# --------------------------------------------------------------------------- #
#  Ground truth
# --------------------------------------------------------------------------- #

def cell_indices(xyz, cfg: Config):
    """Map points to BEV cells.

    Returns ``(i, j, keep)``: row/column of every kept point, plus the mask of
    points that survived the z-range and grid-extent tests. Row/column flips
    are applied. This is the single source of truth for point -> cell mapping,
    shared by label rasterization and point-wise evaluation.
    """
    n = xyz.shape[0]
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=bool)

    u = xyz[:, cfg.col_axis]      # -> column index j
    v = xyz[:, cfg.row_axis]      # -> row index i
    z = xyz[:, 2]

    keep = (z >= cfg.z_min) & (z <= cfg.z_max)
    if not keep.any():
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, keep

    j = np.floor((u - cfg.x_min) / cfg.cell_size).astype(np.int64)
    i = np.floor((v - cfg.y_min) / cfg.cell_size).astype(np.int64)
    keep &= (j >= 0) & (j < cfg.grid_w) & (i >= 0) & (i < cfg.grid_h)

    i, j = i[keep], j[keep]
    if cfg.flip_cols:
        j = (cfg.grid_w - 1) - j
    if cfg.flip_rows:
        i = (cfg.grid_h - 1) - i
    return i, j, keep


def rasterize_masks(xyz, cfg: Config, positive, observed_points=None):
    """Build the BEV occupancy target from per-point boolean masks.

    Parameters
    ----------
    positive         : [N] bool, points that mark a cell as occupied
    observed_points  : [N] bool or None, points that mark a cell as observed
                       (defaults to all points inside the grid)

    Returns
    -------
    occ      : [H, W] float32, 1.0 where a positive point landed
    observed : [H, W] bool, True where an observed point landed
    """
    h, w = cfg.grid_h, cfg.grid_w
    occ = np.zeros((h, w), dtype=np.float32)
    observed = np.zeros((h, w), dtype=bool)

    i, j, keep = cell_indices(xyz, cfg)
    if i.shape[0] == 0:
        return occ, observed

    pos = np.asarray(positive)[keep]
    obs = (np.ones_like(pos) if observed_points is None
           else np.asarray(observed_points)[keep])

    observed[i[obs], j[obs]] = True
    if pos.any():
        occ[i[pos], j[pos]] = 1.0

    # Fill small holes using a fixed metric reach across BEV resolutions.
    occ = dilate_bev_mask(occ, coverage_kernel_size(cfg.cell_size,
                                                    cfg.label_fill_reach_m))

    return occ, observed


def rasterize_occupancy(xyz, intensity, cfg: Config):
    """
    Build the BEV ground-truth mask from the simulator's intensity tags.

    Returns
    -------
    occ      : [H, W] float32, 1.0 where a point with intensity in
               [lo, hi] landed, else 0.0
    observed : [H, W] bool, True where any point landed
    """
    positive = (intensity >= cfg.intensity_lo) & (intensity <= cfg.intensity_hi)
    return rasterize_masks(xyz, cfg, positive)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #

def discover_files(cfg: Config):
    if cfg.dataset == "kitti":
        from .kitti import discover_files as kitti_discover
        return kitti_discover(cfg)
    root = Path(cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    files = sorted(root.glob(cfg.file_glob))
    if not files:
        raise FileNotFoundError(f"no files matched {cfg.file_glob} under {root}")
    return files


def scene_key(path, cfg: Config):
    if cfg.dataset == "kitti":
        from .kitti import sequence_of
        return sequence_of(path)
    if not cfg.split_by_scene:
        return str(path)
    rel = Path(path).relative_to(cfg.data_root)
    return rel.parts[0] if len(rel.parts) > 1 else "root"


def split_files(files, cfg: Config):
    """Split into train/val/test, optionally grouped by scene directory."""
    if cfg.dataset == "kitti":
        from .kitti import split_files as kitti_split
        return kitti_split(files, cfg)

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
        if self.cfg.dataset == "kitti":
            return self._load_sample_kitti(path)
        xyz, intensity = load_cloud(path)

        finite = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
        if not finite.all():
            xyz, intensity = xyz[finite], intensity[finite]

        # Build labels from the full cloud before any point subsampling.
        occ, observed = rasterize_occupancy(xyz, intensity, self.cfg)
        return xyz, intensity, occ, observed

    def _load_sample_kitti(self, path):
        """SemanticKITTI frame: labels come from the .label file, not intensity."""
        from .kitti import load_frame

        xyz, classes = load_frame(path)
        finite = np.isfinite(xyz).all(axis=1)
        if not finite.all():
            xyz, classes = xyz[finite], classes[finite]

        # A cell is positive when a non-ground, non-ignored point lands in it.
        scored = ~np.isin(classes, tuple(self.cfg.kitti_ignore_classes))
        positive = scored & ~_kitti_ground(classes)
        observed_points = scored if self.cfg.kitti_observe_scored_only else None
        occ, observed = rasterize_masks(xyz, self.cfg, positive, observed_points)
        # Intensity is unused by the model; keep a zero vector of equal length.
        return xyz, np.zeros(xyz.shape[0], dtype=np.float32), occ, observed

    def _augment(self, xyz, index):
        """Geometric augs on a copy of xyz. Labels are not modified."""
        if not self.train:
            return xyz

        cfg = self.cfg
        do_shift = cfg.aug_z_shift
        do_invert = cfg.aug_z_invert and cfg.aug_z_invert_prob > 0
        do_rot = cfg.aug_rot_deg > 0 and cfg.aug_rot_prob > 0
        if not do_shift and not do_invert and not do_rot:
            return xyz

        xyz = xyz.copy()
        rng = np.random.default_rng(
            (torch.initial_seed() + index + 1) % (2 ** 31))

        if do_rot and rng.random() < cfg.aug_rot_prob:
            # Rotate about x and y so the model sees slopes it never gets
            # from the mostly flat KITTI sequences.
            for axis in (0, 1):
                if rng.random() >= 0.5:
                    continue
                theta = np.deg2rad(rng.uniform(-cfg.aug_rot_deg, cfg.aug_rot_deg))
                c, s = np.float32(np.cos(theta)), np.float32(np.sin(theta))
                if axis == 0:      # about x: mix y, z
                    y, z = xyz[:, 1].copy(), xyz[:, 2].copy()
                    xyz[:, 1] = c * y - s * z
                    xyz[:, 2] = s * y + c * z
                else:              # about y: mix z, x
                    z, x = xyz[:, 2].copy(), xyz[:, 0].copy()
                    xyz[:, 2] = c * z - s * x
                    xyz[:, 0] = s * z + c * x

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
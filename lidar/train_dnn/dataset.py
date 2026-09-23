"""PLY loading, ground-truth rasterization, and the torch Dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .config import Config


# --------------------------------------------------------------------------- #
#  PLY reading
# --------------------------------------------------------------------------- #

_PLY_DTYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
}


def read_ply(path):
    """
    Minimal binary-little-endian PLY reader.

    Returns {property_name: np.ndarray} for the vertex element. Only the
    vertex element is supported, which is all pc_transform.py writes.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")

        fmt = None
        n_vertex = 0
        props = []
        in_vertex = False

        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF in header")
            line = line.strip()
            if line == b"end_header":
                break
            if line.startswith(b"format"):
                fmt = line.split()[1].decode()
            elif line.startswith(b"element"):
                parts = line.split()
                in_vertex = parts[1] == b"vertex"
                if in_vertex:
                    n_vertex = int(parts[2])
            elif line.startswith(b"property") and in_vertex:
                parts = line.split()
                if parts[1] == b"list":
                    raise ValueError(f"{path}: list properties unsupported")
                ptype, pname = parts[1].decode(), parts[2].decode()
                if ptype not in _PLY_DTYPES:
                    raise ValueError(f"{path}: unsupported type {ptype}")
                props.append((pname, _PLY_DTYPES[ptype]))

        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: expected binary_little_endian, got {fmt}")
        if n_vertex == 0 or not props:
            return {name: np.zeros(0, dtype=dt) for name, dt in props}

        dtype = np.dtype([(name, dt) for name, dt in props])
        raw = fh.read(dtype.itemsize * n_vertex)
        if len(raw) < dtype.itemsize * n_vertex:
            raise ValueError(f"{path}: truncated vertex data")

        arr = np.frombuffer(raw, dtype=dtype, count=n_vertex)
        return {name: np.ascontiguousarray(arr[name]) for name, _ in props}


def load_cloud(path):
    """Load a PLY -> (xyz float32 [N,3], intensity float32 [N])."""
    data = read_ply(path)
    for key in ("x", "y", "z"):
        if key not in data:
            raise ValueError(f"{path}: missing '{key}' property")
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    intensity = (data["intensity"].astype(np.float32)
                 if "intensity" in data
                 else np.zeros(len(xyz), dtype=np.float32))
    return xyz, intensity


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

    # apply 50x50 max pooling to fill in small holes in the occupancy mask
    occ_tensor = torch.from_numpy(occ).unsqueeze(0).unsqueeze(0)
    occ_tensor = F.max_pool2d(occ_tensor, kernel_size=11, stride=1, padding=5)
    occ = occ_tensor.squeeze().numpy()

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
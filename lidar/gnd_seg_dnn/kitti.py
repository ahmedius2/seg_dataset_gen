"""SemanticKITTI (odometry layout) readers, class maps and split helpers.

Expected layout (``data_root`` points at the ``sequences`` directory)::

    <data_root>/<sequence>/velodyne/<frame>.bin
    <data_root>/<sequence>/labels/<frame>.label

``*.bin`` is the raw KITTI scan: little-endian float32, 4 values per point
``(x, y, z, remission)``. ``*.label`` is little-endian uint32 per point where
the low 16 bits hold the semantic class and the high 16 bits the instance id.

The ground-segmentation protocol mirrors the KAIST Ground-Segmentation
Benchmark used by Patchwork++ / TRAVEL:

* ground      -> road, parking, sidewalk, other-ground, lane-marking, terrain
* non-ground  -> every other label
* ignored     -> unlabeled, outlier and (by default) vegetation

The occupancy target of this repository keeps its original polarity: a BEV
cell is *positive* when a non-ground (obstacle) point lands in it.
"""

from pathlib import Path

import numpy as np

# SemanticKITTI label ids (xentropy.yaml / semantic-kitti.yaml).
KITTI_GROUND_CLASSES = (40, 44, 48, 49, 60, 72)
KITTI_VEGETATION_CLASSES = (70, 71)     # 70 vegetation, 71 trunk
KITTI_UNLABELED_CLASSES = (0, 1)        # 0 unlabeled, 1 outlier

# Defaults used by the benchmark: unlabeled/outlier are never scored and
# vegetation is not considered ground nor non-ground.
KITTI_EVAL_IGNORE_CLASSES = (0, 1, 70)


# --------------------------------------------------------------------------- #
#  Reading
# --------------------------------------------------------------------------- #

def load_scan(path):
    """Return ``(xyz float32 [N, 3], remission float32 [N])`` from a .bin scan."""
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"{path}: size {raw.size * 4} B is not a multiple of 16")
    raw = raw.reshape(-1, 4)
    return np.ascontiguousarray(raw[:, :3]), np.ascontiguousarray(raw[:, 3])


def load_labels(path):
    """Return the per-point semantic class (``label & 0xFFFF``) as int32."""
    raw = np.fromfile(str(path), dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.int32)


def labels_for_scan(scan_path):
    """``.../velodyne/000123.bin`` -> ``.../labels/000123.label``."""
    scan_path = Path(scan_path)
    return scan_path.parent.parent / "labels" / (scan_path.stem + ".label")


def load_frame(path):
    """Return ``(xyz [N,3] float32, classes [N] int32)`` for one frame."""
    xyz, _remission = load_scan(path)
    classes = load_labels(labels_for_scan(path))
    if classes.shape[0] != xyz.shape[0]:
        raise ValueError(
            f"{path}: {xyz.shape[0]} points but "
            f"{classes.shape[0]} labels in {labels_for_scan(path)}")
    return xyz, classes


def ground_mask(classes):
    """True where the label is a ground class."""
    return np.isin(classes, KITTI_GROUND_CLASSES)


def scored_mask(classes, ignore_classes=KITTI_EVAL_IGNORE_CLASSES):
    """True where the point participates in the point-wise metrics."""
    return ~np.isin(classes, tuple(ignore_classes))


# --------------------------------------------------------------------------- #
#  Discovery and splits
# --------------------------------------------------------------------------- #

def normalize_sequences(values):
    """Accept ``[0, 1]``, ``["00", "01"]``, ``"00-06"`` or a mixture."""
    if values is None:
        return []
    if isinstance(values, (str, int)):
        values = [values]
    out = []
    for item in values:
        if isinstance(item, int):
            out.append(f"{item:02d}")
            continue
        if isinstance(item, str) and "-" in item and not item.startswith("-"):
            lo, hi = item.split("-", 1)
            out.extend(f"{n:02d}" for n in range(int(lo), int(hi) + 1))
            continue
        out.append(f"{int(str(item)):02d}")
    # Stable dedupe.
    return list(dict.fromkeys(out))


def sequence_of(path):
    """Sequence directory name of a scan path."""
    return Path(path).parent.parent.name


def discover_files(cfg):
    """Every ``.bin`` scan of the configured sequences, sorted per sequence."""
    root = Path(cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    sequences = normalize_sequences(list(cfg.train_sequences) + list(cfg.val_sequences)
                                    + list(cfg.test_sequences))
    if not sequences:
        sequences = sorted(p.name for p in root.iterdir()
                           if (p / "velodyne").is_dir())
    files, missing = [], []
    for seq in sequences:
        scan_dir = root / seq / "velodyne"
        if not scan_dir.is_dir():
            missing.append(seq)
            continue
        files.extend(sorted(scan_dir.glob("*.bin")))
    if missing:
        raise FileNotFoundError(
            f"no velodyne/ directory for sequence(s) {missing} under {root}")
    if not files:
        raise FileNotFoundError(f"no .bin scans found under {root}")
    if cfg.frame_stride and cfg.frame_stride > 1:
        per_seq = {}
        for f in files:
            per_seq.setdefault(sequence_of(f), []).append(f)
        files = [f for seq in sorted(per_seq)
                 for f in per_seq[seq][::cfg.frame_stride]]
    return files


def split_files(files, cfg):
    """Split by explicit sequence lists (never splits a sequence)."""
    def keep(values):
        wanted = set(normalize_sequences(values))
        return [f for f in files if sequence_of(f) in wanted]

    train = keep(cfg.train_sequences)
    val = keep(cfg.val_sequences)
    test = keep(cfg.test_sequences)
    leftover = [f for f in files
                if sequence_of(f) not in set(normalize_sequences(
                    list(cfg.train_sequences) + list(cfg.val_sequences)
                    + list(cfg.test_sequences)))]
    if leftover and not train and not val:
        raise ValueError(
            "no train/val sequence matched the discovered scans; check "
            "train_sequences/val_sequences against data_root")
    return train, val, test

"""NumPy-only static-pillar pipeline pieces used for TIDL / ONNX deployment.

    points -> voxelize (C++) -> decorate (C++) -> PFN (ONNX) -> scatter (NumPy)
           -> BEV backbone (ONNX)

Does not import PyTorch. voxelize/decorate need the C++ extension
(`python setup.py build_ext --inplace`); scatter works without it.
"""

import numpy as np

try:
    from .. import _voxelizer
except ImportError:
    _voxelizer = None


def _require_extension():
    if _voxelizer is None:
        raise ImportError(
            "C++ voxelizer not built; run `python setup.py build_ext --inplace` "
            "in the lidar/ directory")


def voxelize(points, point_cloud_range, voxel_size, grid_size,
             max_points=20, num_threads=4):
    """points (N, 3) xyz -> (voxels (V,P,3), num_points (V,), coords (V,4))."""
    _require_extension()
    pts = np.asarray(points, dtype=np.float32)
    batched = np.empty((pts.shape[0], 4), np.float32)
    batched[:, 0] = 0.0
    batched[:, 1:] = pts[:, :3]
    return _voxelizer.voxelize(
        batched, list(point_cloud_range[:2]), list(voxel_size[:2]),
        [int(grid_size[0]), int(grid_size[1])], int(max_points), int(num_threads))


def decorate(voxels, num_points, coords, point_cloud_range, voxel_size,
             max_pillars, use_abs_xyz=True, with_distance=False, num_threads=4):
    """-> (features (1, F, max_pillars, P) float32, num_valid_pillars)."""
    _require_extension()
    feats, n = _voxelizer.decorate(
        voxels, num_points, coords, list(point_cloud_range[:3]),
        list(voxel_size[:3]), int(max_pillars), bool(use_abs_xyz),
        bool(with_distance), int(num_threads))
    return feats[None], n


def scatter(pillar_features, coords, num_valid, grid_w, grid_h):
    """pillar_features (1, C, max_pillars, 1) + coords (V,4) -> (1, C, H, W)."""
    c = pillar_features.shape[1]
    bev = np.zeros((c, grid_h * grid_w), np.float32)
    if num_valid:
        idx = coords[:num_valid, 2].astype(np.int64) * grid_w + coords[:num_valid, 3]
        bev[:, idx] = pillar_features[0, :, :num_valid, 0]
    return bev.reshape(1, c, grid_h, grid_w)

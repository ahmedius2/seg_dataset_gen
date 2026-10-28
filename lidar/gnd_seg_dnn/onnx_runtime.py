"""NumPy point encoders and CPU ONNX Runtime inference; does not import PyTorch."""

import time
from pathlib import Path
import onnxruntime as ort
import numpy as np
from .model.pillar_ops import scatter, decorate, voxelize

class PillarEncoderNP:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.point_range = data["range"]
        self.voxel = data["voxel"]
        self.nx, self.ny = map(int, data["grid"])
        self.channels = self.w2.shape[1]

    def __call__(self, points):
        ij = np.floor(
            (points[:, :2] - self.point_range[:2]) / self.voxel[:2]
        ).astype(np.int32)
        inside = ((ij[:, 0] >= 0) & (ij[:, 0] < self.nx) &
                  (ij[:, 1] >= 0) & (ij[:, 1] < self.ny))
        points, ij = points[inside], ij[inside]
        dense = np.zeros((self.channels, self.ny * self.nx), np.float32)
        if not len(points):
            return dense.reshape(1, self.channels, self.ny, self.nx)

        keys = ij[:, 1] * self.nx + ij[:, 0]
        order = np.argsort(keys, kind="stable")
        points, ij, keys = points[order], ij[order], keys[order]
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        inverse = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(keys)]))

        counts = np.diff(np.r_[starts, len(keys)])[:, None]
        means = np.add.reduceat(points, starts, axis=0) / counts
        cluster = points - means[inverse]
        center = np.empty_like(points)
        center[:, :2] = points[:, :2] - (
            ij * self.voxel[:2] + self.voxel[:2] / 2 + self.point_range[:2]
        )
        center[:, 2] = points[:, 2] - (
            self.voxel[2] / 2 + self.point_range[2]
        )

        features = np.concatenate([points, cluster, center], axis=1)
        features = np.maximum(features @ self.w1 + self.b1, 0)
        pooled = np.maximum.reduceat(features, starts, axis=0)
        features = np.concatenate([features, pooled[inverse]], axis=1)
        features = np.maximum(features @ self.w2 + self.b2, 0)
        features = np.maximum.reduceat(features, starts, axis=0)
        dense[:, keys[starts]] = features.T
        return dense.reshape(1, self.channels, self.ny, self.nx)


class HeightMapEncoderNP:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.x_min, self.y_min = map(float, data["origin"])
        self.cell = float(data["cell"])
        self.nx, self.ny = map(int, data["grid"])
        self.clip = float(data["clip"])

    def __call__(self, points):
        ix = np.floor((points[:, 0] - self.x_min) / self.cell).astype(np.int64)
        iy = np.floor((points[:, 1] - self.y_min) / self.cell).astype(np.int64)
        inside = ((ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny))
        flat = iy[inside] * self.nx + ix[inside]
        heights = np.clip(points[inside, 2], -self.clip, self.clip) / (2 * self.clip) + 0.5
        result = np.full(self.ny * self.nx, -1.0, np.float32)
        order = np.lexsort((heights, flat))
        result[flat[order]] = heights[order]
        return result.reshape(1, 1, self.ny, self.nx)


class DynamicPillarOnnxModel:
    def __init__(self, model_path, encoder_path, model_variant):

        model_path = Path(model_path)
        encoder_cls = (HeightMapEncoderNP if model_variant == "heightmap"
                       else PillarEncoderNP)
        self.encoder = encoder_cls(encoder_path)
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.last_timings_ms = (0.0, 0.0)  # (encoding, network)

    def __call__(self, points):
        t0 = time.perf_counter()
        bev = self.encoder(points)
        t1 = time.perf_counter()
        logits = self.session.run(
            [self.output_name], {self.input_name: bev}
        )[0][0, 0]
        self.last_timings_ms = ((t1 - t0) * 1000.0,
                                (time.perf_counter() - t1) * 1000.0)
        logits = np.clip(logits, -80.0, 80.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

class StaticPillarOnnxModel:
    """Static-VFE pipeline: C++ voxelize + decorate, PFN ONNX, NumPy scatter,
    BEV backbone ONNX. `export_dir` holds pfn.onnx and bev_net.onnx."""

    def __init__(self, export_dir, cfg, bev_path=None, pfn_path=None,
                 providers=("CPUExecutionProvider",)):

        export_dir = Path(export_dir)
        self.cfg = cfg
        vfe = cfg.vfe_config
        self.abs_xyz = bool(vfe["USE_ABSLOTE_XYZ"])
        self.with_distance = bool(vfe["WITH_DISTANCE"])
        self.range = [cfg.x_min, cfg.y_min, cfg.z_min, cfg.x_max, cfg.y_max, cfg.z_max]
        self.voxel = [cfg.cell_size, cfg.cell_size, cfg.z_max - cfg.z_min]
        self.grid_w = int(round((cfg.x_max - cfg.x_min) / cfg.cell_size))
        self.grid_h = int(round((cfg.y_max - cfg.y_min) / cfg.cell_size))
        self.max_pillars = int(cfg.max_pillars)
        self.max_points = int(cfg.max_points_per_voxel)
        self.threads = int(getattr(cfg, "voxelizer_threads", 4))
        self.pfn = ort.InferenceSession(
            str(pfn_path or export_dir / "pfn.onnx"), providers=list(providers))
        self.bev = ort.InferenceSession(
            str(bev_path or export_dir / "bev_net.onnx"), providers=list(providers))
        self.pfn_in = self.pfn.get_inputs()[0].name
        self.bev_in = self.bev.get_inputs()[0].name
        self.last_timings_ms = (0.0, 0.0)  # (pillar encoding, BEV network)
        self.dropped_pillars = 0

    def __call__(self, points):
        t0 = time.perf_counter()
        vox, npts, coords = voxelize(
            points, self.range, self.voxel, (self.grid_w, self.grid_h),
            self.max_points, self.threads)
        feats, n = decorate(
            vox, npts, coords, self.range, self.voxel, self.max_pillars,
            self.abs_xyz, self.with_distance, self.threads)
        self.dropped_pillars = max(0, len(vox) - self.max_pillars)
        pillar = self.pfn.run(None, {self.pfn_in: feats})[0]
        bev = scatter(pillar, coords, n, self.grid_w, self.grid_h)
        t1 = time.perf_counter()
        logits = self.bev.run(None, {self.bev_in: bev})[0][0, 0]
        self.last_timings_ms = ((t1 - t0) * 1000.0,
                                (time.perf_counter() - t1) * 1000.0)
        logits = np.clip(logits, -80.0, 80.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

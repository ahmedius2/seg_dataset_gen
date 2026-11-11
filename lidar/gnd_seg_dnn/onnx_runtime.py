"""NumPy point encoders and CPU ONNX Runtime inference; does not import PyTorch."""

import time
from pathlib import Path
import onnxruntime as ort
import numpy as np
from .model.pillar_ops import scatter, decorate, pillar_statistics, vfe_feature_opts, voxelize

class PillarEncoderNP:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.point_range = data["range"]
        self.voxel = data["voxel"]
        self.nx, self.ny = map(int, data["grid"])
        self.abs_axes = data["abs_axes"].astype(bool)
        self.rel_axes = data["rel_axes"].astype(bool)
        self.with_distance = bool(data["with_distance"])
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

        components = [points[:, self.abs_axes], cluster, center[:, self.rel_axes]]
        if self.with_distance:
            components.append(np.linalg.norm(points, axis=1, keepdims=True))
        features = np.concatenate(components, axis=1)
        features = np.maximum(features @ self.w1 + self.b1, 0)
        pooled = np.maximum.reduceat(features, starts, axis=0)
        features = np.concatenate([features, pooled[inverse]], axis=1)
        features = np.maximum(features @ self.w2 + self.b2, 0)
        features = np.maximum.reduceat(features, starts, axis=0)
        dense[:, keys[starts]] = features.T
        return dense.reshape(1, self.channels, self.ny, self.nx)


class StatPillarsEncoderNP:
    def __init__(self, npz_path, num_threads=None):
        data = np.load(npz_path)
        if "projection_bias" in data:
            raise ValueError("This StatPillars export uses the old biased projection; "
                             "retrain and export a bias-free model")
        if "projection_weight" in data:
            raise ValueError("This StatPillars export uses the old CPU projection; "
                             "retrain and re-export the model")
        self.x_min, self.y_min = map(float, data["origin"])
        self.cell = float(data["cell"])
        self.nx, self.ny = map(int, data["grid"])
        point_range = data["point_cloud_range"].astype(np.float32)
        self.point_range = point_range
        self.voxel_size = [self.cell, self.cell, float(point_range[5] - point_range[2])]
        self.max_points = int(data["max_points"])
        self.num_threads = int(data["num_threads"]) if num_threads is None else num_threads

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float32)
        batched = np.zeros((len(points), 4), dtype=np.float32)
        batched[:, 1:] = points[:, :3]
        stats, _ = pillar_statistics(
            batched, self.point_range, self.voxel_size, (self.nx, self.ny),
            self.max_points, self.num_threads, 1)
        return stats


class DynamicPillarOnnxModel:
    def __init__(self, model_path, encoder_path, model_variant, num_threads=None,
                 tidl_artifacts=None):

        model_path = Path(model_path)
        encoder_cls = (StatPillarsEncoderNP if model_variant == "statpillars"
                       else PillarEncoderNP)
        self.encoder = (encoder_cls(encoder_path, num_threads)
                        if model_variant == "statpillars" else encoder_cls(encoder_path))
        if tidl_artifacts is None:
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"])
        else:
            self.session = ort.InferenceSession(
                str(model_path), providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
                provider_options=[{"artifacts_folder": str(tidl_artifacts), "debug_level": 0}, {}])
            if "TIDLExecutionProvider" not in self.session.get_providers():
                raise RuntimeError(f"TIDLExecutionProvider failed to initialize for {model_path}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.last_timings_ms = (0.0, 0.0)  # (encoding, network)

    def predict_logits(self, points):
        t0 = time.perf_counter()
        bev = self.encoder(points)
        t1 = time.perf_counter()
        logits = self.session.run(
            [self.output_name], {self.input_name: bev}
        )[0][0, 0]
        self.last_timings_ms = ((t1 - t0) * 1000.0,
                                (time.perf_counter() - t1) * 1000.0)
        return logits

    def __call__(self, points):
        logits = np.clip(self.predict_logits(points), -80.0, 80.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

class StaticPillarOnnxModel:
    """Static-VFE pipeline: C++ voxelize + decorate, PFN ONNX, NumPy scatter,
    BEV backbone ONNX. `export_dir` holds pfn.onnx and bev_net.onnx."""

    def __init__(self, export_dir, cfg, bev_path=None, pfn_path=None,
                 providers=("CPUExecutionProvider",), tidl_artifacts=None,
                 pfn_on_cpu=False):
        """tidl_artifacts: dir from tidl_compile.py (pfn/, bev/); runs both
        nets on the C7x through TIDLExecutionProvider.
        pfn_on_cpu: keep the PFN on the CPU and run only the BEV net on the C7x."""

        export_dir = Path(export_dir)
        self.cfg = cfg
        vfe = cfg.vfe_config
        self.abs_axes, self.rel_axes, self.with_distance = vfe_feature_opts(vfe)
        self.range = [cfg.x_min, cfg.y_min, cfg.z_min, cfg.x_max, cfg.y_max, cfg.z_max]
        self.voxel = [cfg.cell_size, cfg.cell_size, cfg.z_max - cfg.z_min]
        self.grid_w = int(round((cfg.x_max - cfg.x_min) / cfg.cell_size))
        self.grid_h = int(round((cfg.y_max - cfg.y_min) / cfg.cell_size))
        self.max_pillars = int(cfg.max_pillars)
        self.max_points = int(cfg.max_points_per_voxel)
        self.threads = int(getattr(cfg, "voxelizer_threads", 4))
        def session(path, sub):
            if tidl_artifacts is None or (sub == "pfn" and pfn_on_cpu):
                return ort.InferenceSession(str(path), providers=list(providers))
            opts = {"artifacts_folder": str(Path(tidl_artifacts) / sub),
                    "debug_level": 0}
            result = ort.InferenceSession(
                str(path), providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
                provider_options=[opts, {}])
            if "TIDLExecutionProvider" not in result.get_providers():
                raise RuntimeError(f"TIDLExecutionProvider failed to initialize for {path}")
            return result

        self.pfn = session(pfn_path or export_dir / "pfn.onnx", "pfn")
        self.bev = session(bev_path or export_dir / "bev_net.onnx", "bev")
        self.pfn_in = self.pfn.get_inputs()[0].name
        self.bev_in = self.bev.get_inputs()[0].name
        self.last_timings_ms = (0.0, 0.0)  # (pillar encoding, BEV network)
        self.last_enc_breakdown_ms = {}  # voxelize / decorate / pfn / scatter
        self.dropped_pillars = 0

    def predict_logits(self, points):
        t0 = time.perf_counter()
        vox, npts, coords = voxelize(
            points, self.range, self.voxel, (self.grid_w, self.grid_h),
            self.max_points, self.threads)
        ta = time.perf_counter()
        feats, n = decorate(
            vox, npts, coords, self.range, self.voxel, self.max_pillars,
            self.abs_axes, self.rel_axes, self.with_distance, self.threads)
        self.dropped_pillars = max(0, len(vox) - self.max_pillars)
        tb = time.perf_counter()
        pillar = self.pfn.run(None, {self.pfn_in: feats})[0]
        tc = time.perf_counter()
        bev = scatter(pillar, coords, n, self.grid_w, self.grid_h)
        t1 = time.perf_counter()
        self.last_enc_breakdown_ms = {
            "voxelize": (ta - t0) * 1e3, "decorate": (tb - ta) * 1e3,
            "pfn": (tc - tb) * 1e3, "scatter": (t1 - tc) * 1e3}
        logits = self.bev.run(None, {self.bev_in: bev})[0][0, 0]
        self.last_timings_ms = ((t1 - t0) * 1000.0,
                                (time.perf_counter() - t1) * 1000.0)
        return logits

    def __call__(self, points):
        logits = np.clip(self.predict_logits(points), -80.0, 80.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

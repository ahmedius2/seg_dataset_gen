"""NumPy point encoders and CPU ONNX Runtime inference; does not import PyTorch."""

import time
from pathlib import Path

import numpy as np


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


class OnnxRuntimeModel:
    def __init__(self, model_path, encoder_path, model_variant):
        import onnxruntime as ort

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
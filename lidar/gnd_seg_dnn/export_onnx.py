"""Export the lite model for TI AM67A (TIDL).

Splits the network into
  * pillar encoder (dynamic, scatter/unique ops -> not offloadable): BN-folded
    weights saved to <out>/pillar_vfe.npz, run on the A53 cores by PillarEncoderNP
  * dense BEV net (pure convs): <out>/bev_net.onnx, compiled by TIDL for the C7x

Usage:
    python -m gnd_seg_dnn.export_onnx --run-dir dnn_runs_lite
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .train import build_model
from .onnx_runtime import StatPillarsEncoderNP, PillarEncoderNP


def _fold(linear, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = (linear.weight * s[:, None]).t()
    b = bn.bias - bn.running_mean * s
    return w.detach().numpy().astype(np.float32), b.detach().numpy().astype(np.float32)


class PFNExport(torch.nn.Module):
    """Static-VFE PFN as 1x1 convs on (1, F, max_pillars, P), BN folded.

    Only Conv/ReLU/MaxPool/Resize/Concat so TIDL can compile it. Matches
    PillarVFE.pfn_layers (max over all P slots, padded ones included).
    Output: (1, C, max_pillars, 1).
    """

    def __init__(self, vfe, max_points, single_pool=False):
        super().__init__()
        self.single_pool = single_pool
        self.convs = torch.nn.ModuleList()
        self.last = []
        for layer in vfe.pfn_layers:
            lin = layer.linear
            if layer.use_norm:
                w, b = _fold(lin, layer.norm)
                w, b = torch.from_numpy(w).t(), torch.from_numpy(b)
            else:
                w, b = lin.weight.detach(), lin.bias.detach()
            conv = torch.nn.Conv2d(w.shape[1], w.shape[0], 1)
            conv.weight.data = w[:, :, None, None].contiguous()
            conv.bias.data = b
            self.convs.append(conv)
            self.last.append(layer.last_vfe)
        self.max_points = max_points
        if single_pool:
            # One 1xP pool and one broadcast: only valid when MaxPool/Concat/Relu
            # stay on the CPU (hybrid compile), TIDL rejects them.
            self.steps = 0
            return
        if max_points < 1 or max_points & (max_points - 1):
            raise ValueError("max_points_per_voxel must be a power of two for TIDL")
        self.steps = max_points.bit_length() - 1
        # TIDL rejects a 1xP pool and non-square Resize: use log2(P) 1x2 pools
        # and width-doubling Concats, which are exactly equivalent.
        self.pool = torch.nn.MaxPool2d((1, 2), stride=(1, 2))

    def _pool(self, x):
        if self.single_pool:
            return torch.amax(x, dim=3, keepdim=True)
        for _ in range(self.steps):
            x = self.pool(x)
        return x

    def _up(self, x):
        if self.single_pool:
            return x.expand(-1, -1, -1, self.max_points)
        for _ in range(self.steps):
            # x is post-ReLU so relu(x) == x; it keeps the Concat inputs distinct
            # (TIDL fails to topologically sort a node fed twice by one tensor).
            x = torch.cat([x, torch.relu(x)], dim=3)
        return x

    def forward(self, x):
        for conv, last in zip(self.convs, self.last):
            x = torch.relu(conv(x))
            pooled = self._pool(x)
            x = pooled if last else torch.cat([x, self._up(pooled)], dim=1)
        return x


def _export_static(cfg, model, out, opset, single_pool=False):
    """Static VFE: voxelize/decorate (C++), PFN (ONNX), scatter (NumPy)."""
    from .model.pillar_ops import scatter, decorate, voxelize
    import onnxruntime as ort

    vfe = model.vfe
    p = cfg.max_points_per_voxel
    pfn = PFNExport(vfe, p, single_pool).eval()
    f = vfe.pfn_layers[0].linear.in_features
    dummy = torch.zeros(1, f, cfg.max_pillars, p)
    torch.onnx.export(pfn, dummy, str(out / "pfn.onnx"), opset_version=opset,
                      input_names=["pillar_features_in"],
                      output_names=["pillar_features"], do_constant_folding=True,
                      dynamo=False)
    c = cfg.scatter_config["NUM_BEV_FEATURES"]
    bev_dummy = torch.zeros(1, c, cfg.grid_h, cfg.grid_w)
    torch.onnx.export(model.backbone, bev_dummy, str(out / "bev_net.onnx"),
                      opset_version=opset, input_names=["bev_in"],
                      output_names=["logits"], do_constant_folding=True,
                      dynamo=False)
    print(f"saved {out}/pfn.onnx (input {tuple(dummy.shape)}) and "
          f"{out}/bev_net.onnx (input {tuple(bev_dummy.shape)})")

    # Sanity check of the whole deployed pipeline against the torch model.
    pts = np.random.uniform([-6, -25, -2], [6, 25, 2], (20000, 3)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.cat([torch.zeros(len(pts), 1), torch.from_numpy(pts)], 1))
    vox, npts, coords = voxelize(
        pts, cfg.point_cloud_range, cfg.pillar_dims, cfg.grid_size, p,
        cfg.voxelizer_threads)
    feats, n = decorate(
        vox, npts, coords, cfg.point_cloud_range, cfg.pillar_dims, cfg.max_pillars,
        vfe.abs_axes, vfe.rel_axes, vfe.with_distance, cfg.voxelizer_threads)
    out_pfn = ort.InferenceSession(str(out / "pfn.onnx"),
                                   providers=["CPUExecutionProvider"]).run(
        None, {"pillar_features_in": feats})[0]
    bev = scatter(out_pfn, coords, n, cfg.grid_w, cfg.grid_h)
    with torch.no_grad():
        got = model.backbone(torch.from_numpy(bev)).squeeze(1)
    print(f"pillars used {len(vox)} (max_pillars {cfg.max_pillars})")
    print("max |logit diff| deployed-pipeline-vs-torch:", (ref - got).abs().max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="training run directory")
    ap.add_argument("--single-pool", action="store_true",
                    help="PFN with one ReduceMax + Expand instead of log2(P) pools; only for "
                         "hybrid compile (--pfn-deny-types MaxPool,Concat,Relu,ReduceMax,Expand)")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--max-pillars", type=int, default=None,
                    help="fixed pillar count of the static-VFE PFN (default: from config)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_yaml(run_dir / "config.yaml")
    cfg.assert_model_compatible(Config.from_dict(ckpt["config"]))
    assert cfg.model_variant in ("lite", "statpillars"), "checkpoint is not an embedded variant"

    if args.max_pillars is not None:
        if args.max_pillars < 1:
            ap.error("--max-pillars must be positive")
        cfg.max_pillars = args.max_pillars
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out = run_dir / "onnx"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        ap.error(f"export directory is not empty: {out}; move or remove it before exporting "
                 "to avoid retaining stale ONNX or TIDL artifacts")
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out / "config.yaml")

    if cfg.model_variant == "lite" and cfg.vfe_type == "static":
        _export_static(cfg, model, out, args.opset, args.single_pool)
        return

    if cfg.model_variant == "statpillars":
        np.savez(out / "pillar_vfe.npz", origin=np.array([cfg.x_min, cfg.y_min], np.float32),
                 cell=np.float32(cfg.cell_size),
                 point_cloud_range=np.array(cfg.point_cloud_range, np.float32),
                 grid=np.array([cfg.grid_w, cfg.grid_h], np.int32),
                 max_points=np.int32(cfg.max_points_per_voxel),
                 num_threads=np.int32(cfg.voxelizer_threads))
        c = 15
    else:
        p1, p2 = model.vfe.pfn_layers
        w1, b1 = _fold(p1.linear, p1.norm)
        w2, b2 = _fold(p2.linear, p2.norm)
        np.savez(out / "pillar_vfe.npz", w1=w1, b1=b1, w2=w2, b2=b2,
                 range=np.array(cfg.point_cloud_range, np.float32),
                 voxel=np.array(cfg.pillar_dims, np.float32),
                 grid=np.array([cfg.grid_w, cfg.grid_h], np.int32),
                 abs_axes=np.array(model.vfe.abs_axes, dtype=bool),
                 rel_axes=np.array(model.vfe.rel_axes, dtype=bool),
                 with_distance=np.array(model.vfe.with_distance))
        c = cfg.scatter_config["NUM_BEV_FEATURES"]
    dummy = torch.zeros(1, c, cfg.grid_h, cfg.grid_w)
    torch.onnx.export(model.backbone, dummy, str(out / "bev_net.onnx"),
                      opset_version=args.opset, input_names=["bev_in"],
                      output_names=["logits"], do_constant_folding=True,
                      dynamo=False)
    print(f"saved {out}/bev_net.onnx and pillar_vfe.npz; input shape {tuple(dummy.shape)}")

    # Sanity check: NumPy encoder + dense net vs. full torch model.
    pts = np.random.uniform([-6, -25, -2], [6, 25, 2], (20000, 3)).astype(np.float32)
    enc_cls = StatPillarsEncoderNP if cfg.model_variant == "statpillars" else PillarEncoderNP
    enc = enc_cls(out / "pillar_vfe.npz")
    if cfg.model_variant == "statpillars":
        model.encoder.num_threads = 1
        enc.num_threads = 1
    with torch.no_grad():
        ref = model(torch.cat([torch.zeros(len(pts), 1), torch.from_numpy(pts)], 1))
        got = model.backbone(torch.from_numpy(enc(pts))).squeeze(1)
    print("max |logit diff| numpy-vs-torch:", (ref - got).abs().max().item())


if __name__ == "__main__":
    main()

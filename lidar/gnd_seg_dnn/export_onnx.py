"""Export the lite model for TI AM67A (TIDL).

Splits the network into
  * pillar encoder (dynamic, scatter/unique ops -> not offloadable): BN-folded
    weights saved to <out>/pillar_vfe.npz, run on the A53 cores by PillarEncoderNP
  * dense BEV net (pure convs): <out>/bev_net.onnx, compiled by TIDL for the C7x

Usage:
    python -m gnd_seg_dnn.export_onnx --ckpt dnn_runs_lite/best.pt --out export_lite
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .train import build_model
from .onnx_runtime import HeightMapEncoderNP, PillarEncoderNP


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

    def __init__(self, vfe, max_points):
        super().__init__()
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
        self.pool = torch.nn.MaxPool2d((1, max_points))
        self.up = torch.nn.Upsample(scale_factor=(1, max_points), mode="nearest")

    def forward(self, x):
        for conv, last in zip(self.convs, self.last):
            x = torch.relu(conv(x))
            pooled = self.pool(x)
            x = pooled if last else torch.cat([x, self.up(pooled)], dim=1)
        return x


def _export_static(cfg, model, out, opset):
    """Static VFE: voxelize/decorate (C++), PFN (ONNX), scatter (NumPy)."""
    from .model.pillar_ops import scatter, decorate, voxelize
    import onnxruntime as ort

    vfe = model.vfe
    p = cfg.max_points_per_voxel
    pfn = PFNExport(vfe, p).eval()
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
        vfe.use_absolute_xyz, vfe.with_distance, cfg.voxelizer_threads)
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="export_lite")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--max-pillars", type=int, default=None,
                    help="fixed pillar count of the static-VFE PFN (default: from config)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["config"])
    for k in ("grid_h", "grid_w", "grid_size"):
        cfg_dict.pop(k, None)
    cfg = Config(**cfg_dict)
    cfg.__post_init__()
    assert cfg.model_variant in ("lite", "heightmap"), "checkpoint is not an embedded variant"

    if args.max_pillars:
        cfg.max_pillars = args.max_pillars
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")

    if cfg.model_variant == "lite" and cfg.vfe_type == "static":
        _export_static(cfg, model, out, args.opset)
        return

    if cfg.model_variant == "heightmap":
        np.savez(out / "pillar_vfe.npz", origin=np.array([cfg.x_min, cfg.y_min], np.float32),
                 cell=np.float32(cfg.cell_size), clip=np.float32(cfg.height_clip),
                 grid=np.array([cfg.grid_w, cfg.grid_h], np.int32))
        c = 1
    else:
        p1, p2 = model.vfe.pfn_layers
        w1, b1 = _fold(p1.linear, p1.norm)
        w2, b2 = _fold(p2.linear, p2.norm)
        np.savez(out / "pillar_vfe.npz", w1=w1, b1=b1, w2=w2, b2=b2,
                 range=np.array(cfg.point_cloud_range, np.float32),
                 voxel=np.array(cfg.pillar_dims, np.float32),
                 grid=np.array([cfg.grid_w, cfg.grid_h], np.int32))
        c = cfg.scatter_config["NUM_BEV_FEATURES"]
    dummy = torch.zeros(1, c, cfg.grid_h, cfg.grid_w)
    torch.onnx.export(model.backbone, dummy, str(out / "bev_net.onnx"),
                      opset_version=args.opset, input_names=["bev_in"],
                      output_names=["logits"], do_constant_folding=True,
                      dynamo=False)
    print(f"saved {out}/bev_net.onnx and pillar_vfe.npz; input shape {tuple(dummy.shape)}")

    # Sanity check: NumPy encoder + dense net vs. full torch model.
    pts = np.random.uniform([-6, -25, -2], [6, 25, 2], (20000, 3)).astype(np.float32)
    ref = model(torch.cat([torch.zeros(len(pts), 1), torch.from_numpy(pts)], 1))
    enc_cls = HeightMapEncoderNP if cfg.model_variant == "heightmap" else PillarEncoderNP
    enc = enc_cls(out / "pillar_vfe.npz")
    with torch.no_grad():
        got = model.backbone(torch.from_numpy(enc(pts))).squeeze(1)
    print("max |logit diff| numpy-vs-torch:", (ref - got).abs().max().item())


if __name__ == "__main__":
    main()

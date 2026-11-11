"""Compile static-pillar or StatPillars BEV networks for AM67A C7x.

Run on an x86 PC (not on the board) inside the edgeai-tidl-tools OSRT
environment (onnxruntime with TIDLCompilationProvider, TIDL_TOOLS_PATH set):

  python -m gnd_seg_dnn.tidl_compile --run-dir run \
      --calib-dir /path/to/clouds

Calibration clouds are the project's binary-little-endian .ply files, or
.npy (N,3|4 float32, xyz first) / .bin (N,4 float32) files.
Compiled artifacts are stored in run/onnx/tidl_artifacts.
"""

import argparse
import os
import multiprocessing as mp
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort

from .model.pillar_ops import decorate, scatter, vfe_feature_opts, voxelize
from .onnx_runtime import StatPillarsEncoderNP
from .config import load_yaml_mapping
from .ply_io import load_cloud as load_ply_cloud

os.environ["TIDL_RT_PERFSTATS"] = "1"

def load_cloud(path):
    if path.suffix == ".npy":
        pts = np.load(path)
        return np.ascontiguousarray(pts[:, :3], np.float32)
    if path.suffix == ".ply":
        pts, _ = load_ply_cloud(path)
        return np.ascontiguousarray(pts, np.float32)
    if path.suffix == ".bin":
        pts = np.fromfile(path, np.float32).reshape(-1, 4)
        return np.ascontiguousarray(pts[:, :3], np.float32)
    raise ValueError(f"unsupported calibration cloud format: {path}")


def load_cfg(export_dir):
    cfg = SimpleNamespace(**load_yaml_mapping(export_dir / "config.yaml"))
    cfg.grid_w = int(round((cfg.x_max - cfg.x_min) / cfg.cell_size))
    cfg.grid_h = int(round((cfg.y_max - cfg.y_min) / cfg.cell_size))
    return cfg


def make_calibration(cfg, pfn_sess, clouds):
    """Run the real pre-processing so both nets calibrate on true activations."""
    vfe = cfg.vfe_config
    rng = [cfg.x_min, cfg.y_min, cfg.z_min, cfg.x_max, cfg.y_max, cfg.z_max]
    vox_size = [cfg.cell_size, cfg.cell_size, cfg.z_max - cfg.z_min]
    threads = int(getattr(cfg, "voxelizer_threads", 4))
    in_name = pfn_sess.get_inputs()[0].name
    pfn_in, bev_in = [], []
    for path in clouds:
        vox, npts, coords = voxelize(
            load_cloud(path), rng, vox_size, (cfg.grid_w, cfg.grid_h),
            int(cfg.max_points_per_voxel), threads)
        feats, n = decorate(
            vox, npts, coords, rng, vox_size, int(cfg.max_pillars),
            *vfe_feature_opts(vfe), threads)
        pillar = pfn_sess.run(None, {in_name: feats})[0]
        pfn_in.append(feats)
        bev_in.append(scatter(pillar, coords, n, cfg.grid_w, cfg.grid_h))
    return pfn_in, bev_in


def make_statpillars_calibration(export_dir, clouds):
    """Run the deployed CPU encoder to generate BEV calibration inputs."""
    encoder = StatPillarsEncoderNP(export_dir / "pillar_vfe.npz")
    return [encoder(load_cloud(path)) for path in clouds]


def compile_model(onnx_path, art_dir, inputs, args, bits=None, extra_opts=None):
    art_dir.mkdir(parents=True, exist_ok=True)
    # TIDL import needs every tensor shape in the graph; torch export omits them.
    inferred = Path(tempfile.mkdtemp()) / f"{onnx_path.stem}.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path))), str(inferred))
    onnx_path = inferred
    name = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    n = len(inputs)
    opts = {
        "tidl_tools_path": args.tidl_tools,
        "artifacts_folder": str(art_dir),
        "tensor_bits": bits or args.bits,
        "accuracy_level": 1,
        "debug_level": args.debug_level,                     # <-- was 3
        "advanced_options:calibration_frames": n,
        "advanced_options:calibration_iterations": min(args.calib_iters, n),  # <-- clamp
    }
    opts.update(extra_opts or {})
    so = ort.SessionOptions()
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=so,
        providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
        provider_options=[opts, {}])
    # TI's ONNX Runtime 1.15 reports the compilation provider as
    # TIDLExecutionProvider after session initialization.
    if not ({"TIDLCompilationProvider", "TIDLExecutionProvider"}
            & set(sess.get_providers())):
        raise RuntimeError(f"TIDLCompilationProvider failed to initialize for {onnx_path}")
    for x in inputs:
        sess.run(None, {name: x})
    print(f"compiled {onnx_path.name} -> {art_dir} ({n} calibration frames)")


def _compile_worker(*a):
    compile_model(*a)


def compile_isolated(*a):
    """TIDL's emulated device memory is not released when a session is freed, so a
    second compile in the same process dies with 'partially present on the device'."""
    p = mp.get_context("spawn").Process(target=_compile_worker, args=a)
    p.start()
    p.join()
    if p.exitcode:
        raise SystemExit(f"TIDL compile of {a[0].name} failed (exit code {p.exitcode})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="training run directory containing onnx/")
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--num-frames", type=int, default=20)
    ap.add_argument("--bits", type=int, default=8, choices=(8, 16))
    ap.add_argument("--pfn-bits", type=int, default=16, choices=(8, 16),
                    help="PFN precision. Its output is post-ReLU (unsigned); at 8 bit the "
                         "TIDL import hangs in TIDL_runtimesPostProcessNet")
    ap.add_argument("--skip-pfn", action="store_true",
                    help="do not compile the PFN (run it with --pfn-on-cpu)")
    ap.add_argument("--pfn-deny-types", default="",
                    help="hybrid PFN: comma-separated ONNX op types kept on the CPU, e.g. "
                         "MaxPool,Concat (the rest runs on the C7x as separate TIDL subgraphs)")
    ap.add_argument("--pfn-deny-names", default="",
                    help="hybrid PFN: comma-separated ONNX node names kept on the CPU")
    ap.add_argument("--calib-iters", type=int, default=10)
    ap.add_argument("--debug-level", type=int, default=0)
    ap.add_argument("--tidl-tools", default=os.environ.get("TIDL_TOOLS_PATH"))
    args = ap.parse_args()
    if not args.tidl_tools:
        ap.error("set --tidl-tools or TIDL_TOOLS_PATH (edgeai-tidl-tools/tidl_tools)")

    export_dir = Path(args.run_dir) / "onnx"
    cfg = load_cfg(export_dir)
    statpillars = cfg.model_variant in ("statpillars", "heightmap")
    if not statpillars and (cfg.model_variant != "lite" or cfg.vfe_type != "static"):
        ap.error("TIDL compilation requires a static lite or StatPillars model")
    if statpillars and (args.skip_pfn or args.pfn_deny_types or args.pfn_deny_names):
        ap.error("PFN options do not apply to StatPillars; its encoder runs on the CPU")
    if args.num_frames < 1 or args.calib_iters < 1:
        ap.error("--num-frames and --calib-iters must be positive")
    out = export_dir / "tidl_artifacts"
    for name in (("bev",) if statpillars or args.skip_pfn else ("pfn", "bev")):
        target = out / name
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            ap.error(f"artifact directory already exists and is not empty: {target}; "
                     "move or remove it before recompiling")
    clouds = sorted(p for p in Path(args.calib_dir).iterdir()
                    if p.suffix in (".ply", ".npy", ".bin"))[:args.num_frames]
    if not clouds:
        ap.error(f"no .ply, .npy, or .bin clouds in {args.calib_dir}")

    bev_path = export_dir / "bev_net.onnx"
    if statpillars:
        bev_in = make_statpillars_calibration(export_dir, clouds)
        compile_isolated(bev_path, out / "bev", bev_in, args)
        return

    pfn_path = export_dir / "pfn.onnx"
    pfn_sess = ort.InferenceSession(str(pfn_path),
                                    providers=["CPUExecutionProvider"])
    pfn_in, bev_in = make_calibration(cfg, pfn_sess, clouds)
    if not args.skip_pfn:
        extra = {}
        if args.pfn_deny_types:
            extra["deny_list:layer_type"] = args.pfn_deny_types
        if args.pfn_deny_names:
            extra["deny_list:layer_name"] = args.pfn_deny_names
        if extra:
            extra["advanced_options:add_data_convert_ops"] = 0
        compile_isolated(pfn_path, out / "pfn", pfn_in, args, args.pfn_bits, extra)
    compile_isolated(bev_path, out / "bev", bev_in, args)


if __name__ == "__main__":
    main()

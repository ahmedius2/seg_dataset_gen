"""Run-directory loading and CLI options shared by ROS inference."""

import argparse
from pathlib import Path

from .config import Config, load_yaml_mapping


def inference_parser():
    parser = argparse.ArgumentParser(description="Run ROS 2 occupancy inference")
    parser.add_argument("--run-dir", required=True, help="directory containing config.yaml, best.pt and onnx/")
    parser.add_argument("--backend", choices=("torch", "onnx", "tidl"), default="torch")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Torch device")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="enable or disable CUDA mixed precision for Torch (default: enabled on CUDA)")
    parser.add_argument("--voxelizer-threads", type=int,
                        help="override the static or StatPillars CPU voxelizer thread count")
    parser.add_argument("--pfn-on-cpu", action="store_true",
                        help="with TIDL, run the PFN on the CPU")
    parser.add_argument("--world-frame", default="scene_0000_0")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="occupancy probability threshold")
    parser.add_argument("--no-timings", action="store_true")
    return parser


def parse_inference_args(argv):
    parser = inference_parser()
    args = parser.parse_args(argv)
    if args.pfn_on_cpu and args.backend != "tidl":
        parser.error("--pfn-on-cpu requires --backend tidl")
    if args.device != "auto" and args.backend != "torch":
        parser.error("--device applies only to the torch backend")
    if args.amp is not None and args.backend != "torch":
        parser.error("--amp and --no-amp apply only to the torch backend")
    if args.voxelizer_threads is not None and args.voxelizer_threads < 1:
        parser.error("--voxelizer-threads must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    return args


def load_run_model(args):
    run = Path(args.run_dir)
    cfg = Config.from_yaml(run / "config.yaml")
    if args.voxelizer_threads is not None:
        if cfg.model_variant != "statpillars" and cfg.vfe_type != "static":
            raise ValueError("--voxelizer-threads requires a static or StatPillars model")
        cfg.voxelizer_threads = args.voxelizer_threads
    if args.backend == "torch":
        import torch
        from .train import build_model

        device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        cfg.assert_model_compatible(Config.from_dict(checkpoint["config"]))
        model = build_model(cfg)
        model.load_state_dict(checkpoint["model"], strict=True)
        device = torch.device(device_name)
        return model.to(device).eval(), cfg, device

    import onnxruntime as ort
    from .onnx_runtime import DynamicPillarOnnxModel, StaticPillarOnnxModel

    onnx_dir = run / "onnx"
    exported = load_yaml_mapping(onnx_dir / "config.yaml")
    export_cfg = Config.from_dict(exported)
    cfg.assert_model_compatible(export_cfg)
    if args.voxelizer_threads is not None:
        export_cfg.voxelizer_threads = args.voxelizer_threads
    export_cfg.max_pillars = exported["max_pillars"]
    if type(export_cfg.max_pillars) is not int or export_cfg.max_pillars < 1:
        raise ValueError("Exported max_pillars must be a positive integer")
    bev = onnx_dir / "bev_net.onnx"
    if not bev.is_file():
        raise FileNotFoundError(f"ONNX model not found: {bev}")
    if export_cfg.model_variant not in ("lite", "statpillars"):
        raise ValueError("ONNX backend requires a lite or statpillars model")
    if export_cfg.vfe_type == "static" and export_cfg.model_variant == "lite":
        pfn = onnx_dir / "pfn.onnx"
        if not pfn.is_file():
            raise FileNotFoundError(f"PFN model not found: {pfn}")
        artifacts = None
        if args.backend == "tidl":
            if "TIDLExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("TIDLExecutionProvider is not available in ONNX Runtime")
            artifacts = onnx_dir / "tidl_artifacts"
            for name in (("bev",) if args.pfn_on_cpu else ("pfn", "bev")):
                path = artifacts / name
                if not path.is_dir() or not any(path.iterdir()):
                    raise FileNotFoundError(f"TIDL artifacts missing or empty: {path}")
        model = StaticPillarOnnxModel(onnx_dir, export_cfg, tidl_artifacts=artifacts,
                                      pfn_on_cpu=args.pfn_on_cpu)
        return model, export_cfg, "TIDL" if artifacts else "CPU (ONNX Runtime)"

    artifacts = None
    if args.backend == "tidl":
        if export_cfg.model_variant != "statpillars":
            raise ValueError("TIDL inference requires a static lite or StatPillars model")
        if args.pfn_on_cpu:
            raise ValueError("--pfn-on-cpu does not apply to StatPillars")
        if "TIDLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("TIDLExecutionProvider is not available in ONNX Runtime")
        artifacts = onnx_dir / "tidl_artifacts" / "bev"
        if not artifacts.is_dir() or not any(artifacts.iterdir()):
            raise FileNotFoundError(f"TIDL artifacts missing or empty: {artifacts}")
    encoder = onnx_dir / "pillar_vfe.npz"
    if not encoder.is_file():
        raise FileNotFoundError(f"ONNX encoder not found: {encoder}")
    return DynamicPillarOnnxModel(bev, encoder, export_cfg.model_variant,
                                  export_cfg.voxelizer_threads, artifacts), export_cfg, (
                                      "TIDL" if artifacts else "CPU (ONNX Runtime)")

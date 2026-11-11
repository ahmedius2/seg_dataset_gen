"""Central configuration for the BEV occupancy training pipeline."""

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
import yaml


class EasyDict(dict):
    """Dict that also supports attribute access, used by the model_cfg blocks."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def load_yaml_mapping(path):
    path = Path(path)
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"Configuration must be a YAML file: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


@dataclass
class Config:
    # ---- Data ----
    data_root: str = "/home/ubuntu/shared/dnn_dataset"
    file_glob: str = "**/*.ply"
    val_fraction: float = 0.20
    test_fraction: float = 0.00
    split_seed: int = 1234
    # Split by scene directory so frames from one scene never leak across splits.
    split_by_scene: bool = True

    # ---- Ground truth ----
    intensity_lo: float = 190.0
    intensity_hi: float = 210.0
    # Cells no point touches are excluded from the loss.
    ignore_unobserved: bool = True

    # ---- BEV grid geometry (meters) ----
    x_min: float = -6.4
    x_max: float = 6.4
    y_min: float = -25.6
    y_max: float = 25.6
    z_min: float = -10.0 # ignore z, its using pillars
    z_max: float = 10.0
    point_cloud_range: list[float] = field(default_factory=lambda: [-6.4, -25.6, -10.0, 6.4, 25.6, 10.0])
    cell_size: float = 0.8          # 0.8 m x 0.8 m = 0.64 m^2 per cell
    # z spans the full z range so every point falls into a single z pillar bucket
    pillar_dims: list[float] = field(default_factory=lambda: [0.8, 0.8, 20.0])
    # Which cloud axis feeds the grid column (j) and row (i). 0=x, 1=y, 2=z
    col_axis: int = 0
    row_axis: int = 1
    flip_rows: bool = False
    flip_cols: bool = False
    num_point_features = 3

    # ---- Point sampling ----
    #max_points: int = 60000
    #min_points: int = 16
    # Keep each decoded cloud and rasterized label in RAM after first access.
    # Set False to reread and preprocess PLY files on every access.
    cache_in_memory: bool = True

    # ---- Train-time augmentations (input points only; GT is unchanged) ----
    # Uniform random translation of every point along z, in meters.
    aug_z_shift: bool = True
    aug_z_shift_min: float = -1.0
    aug_z_shift_max: float = 1.0
    # Randomly negate every point's z coordinate (z -> -z).
    aug_z_invert: bool = True
    aug_z_invert_prob: float = 0.5

    # ---- Optimization ----
    epochs: int = 100
    batch_size: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    warmup_epochs: int = 2
    scheduler: str = "cosine"       # "cosine" | "step" | "none"
    amp: bool = True
    num_workers: int = 4
    seed: int = 0

    # ---- Loss ----
    loss_type: str = "focal"        # "focal" | "bce" | "soft_iou"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    pos_weight: float = 1.0         # only used by "bce"
    dice_weight: float = 0.0        # optional auxiliary soft-IoU term

    # ---- Output ----
    out_dir: str = "dnn_runs"
    log_every: int = 20
    save_every: int = 5
    eval_threshold: float = 0.5
    val_visualizations: int = 4
    save_visualizations: bool = True
    point_stats: bool = True
    # Deprecated and unused: coverage reach is now fixed in meters via
    # bev_ops.coverage_kernel_size(cell_size). Kept so old checkpoints load.
    pred_coverage_kernel: int = 11

    # Stop when val IoU hasn't improved by min_delta for this many epochs (0 = off).
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-3

    # Heightmap variant: heights are clipped to +-height_clip metres before scaling.
    height_clip: float = 5.0

    # "dynamic" = DynamicPillarVFE (all points), "static" = PillarVFE (at most
    # max_points_per_voxel points per pillar, extras dropped).
    vfe_type: str = "dynamic"
    max_points_per_voxel: int = 32
    # Fixed pillar count of the exported PFN (TIDL needs static shapes);
    # pillars beyond it are dropped at inference. Only utilized when exporting.
    max_pillars: int = 4096
    voxelizer_threads: int = 4      # C++ voxelizer threads (static VFE on CPU)

    # "full" = original OpenPCDet-style net, "lite" = TIDL/embedded-friendly net
    model_variant: str = "lite"

    # ---- Derived ----
    grid_h: int = field(init=False, default=0)
    grid_w: int = field(init=False, default=0)
    grid_size: list[int] = field(init=False, default_factory=lambda: [0, 0])

    vfe_config: dict = field(default_factory=lambda: EasyDict({
        "WITH_DISTANCE_FROM_PC_ORIG": False,
        "ABS_POINT_COORDS": [0, 1, 2],
        "REL_POINT_COORDS": [0, 1, 2],
        "USE_NORM": True,
        "NUM_FILTERS": [ 16, 16 ]}))

    scatter_config: dict = field(default_factory=lambda: EasyDict({
        "NUM_BEV_FEATURES": 16}))

    backbone_2d_config: dict = field(default_factory=lambda: EasyDict({
        "LAYER_NUMS": [3, 5, 5],
        "LAYER_STRIDES": [2, 2, 2],
        "NUM_FILTERS": [64, 128, 256],
        "UPSAMPLE_STRIDES": [2, 4, 8],
        "NUM_UPSAMPLE_FILTERS": [128, 128, 128]}))

    def __post_init__(self):
        self.vfe_config = EasyDict(self.vfe_config)
        self.scatter_config = EasyDict(self.scatter_config)
        self.backbone_2d_config = EasyDict(self.backbone_2d_config)
        allowed_vfe = {"WITH_DISTANCE_FROM_PC_ORIG", "ABS_POINT_COORDS",
                       "REL_POINT_COORDS", "USE_NORM", "NUM_FILTERS"}
        unknown = self.vfe_config.keys() - allowed_vfe
        if unknown:
            raise ValueError(f"Unknown vfe_config fields: {sorted(unknown)}")
        for name in ("WITH_DISTANCE_FROM_PC_ORIG", "USE_NORM"):
            if not isinstance(self.vfe_config.get(name), bool):
                raise ValueError(f"vfe_config.{name} must be a boolean")
        for name in ("ABS_POINT_COORDS", "REL_POINT_COORDS"):
            axes = self.vfe_config.get(name)
            if (not isinstance(axes, list)
                    or any(type(a) is not int or a not in (0, 1, 2) for a in axes)
                    or len(set(axes)) != len(axes)):
                raise ValueError(f"vfe_config.{name} must contain unique axis indices 0, 1, 2")
        filters = self.vfe_config.get("NUM_FILTERS")
        if (not isinstance(filters, list) or not filters
                or any(type(n) is not int or n <= 0 for n in filters)
                or any(n % 2 for n in filters[:-1])):
            raise ValueError("vfe_config.NUM_FILTERS must contain positive widths; "
                             "non-final widths must be even")
        if type(self.max_points_per_voxel) is not int or self.max_points_per_voxel < 1:
            raise ValueError("max_points_per_voxel must be a positive integer")
        if (self.model_variant != "heightmap"
                and self.scatter_config.get("NUM_BEV_FEATURES") != filters[-1]):
            raise ValueError("scatter_config.NUM_BEV_FEATURES must match the final PFN width")
        self.grid_w = int(round((self.x_max - self.x_min) / self.cell_size))
        self.grid_h = int(round((self.y_max - self.y_min) / self.cell_size))
        if self.grid_w <= 0 or self.grid_h <= 0:
            raise ValueError("Invalid BEV extent / cell_size")
        self.max_pillars = self.grid_h * self.grid_w
        # Extents must be whole multiples of cell_size, otherwise the grid
        # silently covers a different range than x/y_min..max.
        for name, extent, n in (("x", self.x_max - self.x_min, self.grid_w),
                                ("y", self.y_max - self.y_min, self.grid_h)):
            if abs(n * self.cell_size - extent) > 1e-6:
                raise ValueError(
                    f"{name} extent {extent} m is not a multiple of "
                    f"cell_size {self.cell_size}")
        # The backbones downsample 8x and upsample back; other sizes break
        # the output shape.
        if self.grid_w % 8 or self.grid_h % 8:
            raise ValueError(
                f"BEV grid {self.grid_w}x{self.grid_h} must be divisible by 8 "
                f"(cell_size={self.cell_size})")
        if self.vfe_type not in ("dynamic", "static"):
            raise ValueError("vfe_type must be 'dynamic' or 'static'")
        if self.col_axis == self.row_axis:
            raise ValueError("col_axis and row_axis must differ")

        self.grid_size = [self.grid_w, self.grid_h, 1]

        # Single source of truth: pillar geometry follows cell_size and the extents.
        self.pillar_dims = [self.cell_size, self.cell_size, self.z_max - self.z_min]
        self.point_cloud_range = [self.x_min, self.y_min, self.z_min,
                                  self.x_max, self.y_max, self.z_max]

    @classmethod
    def lite(cls, **kw):
        cfg = cls(model_variant="lite", **kw)
        cfg.vfe_config = EasyDict({
            "WITH_DISTANCE_FROM_PC_ORIG": False,
            "ABS_POINT_COORDS": [0, 1, 2], "REL_POINT_COORDS": [0, 1, 2],
            "USE_NORM": True, "NUM_FILTERS": [32, 32]})
        cfg.scatter_config = EasyDict({"NUM_BEV_FEATURES": 32})
        return cfg

    @classmethod
    def heightmap(cls, **kw):
        cfg = cls(model_variant="heightmap", **kw)
        cfg.scatter_config = EasyDict({"NUM_BEV_FEATURES": 1})
        return cfg

    def to_yaml(self, path):
        def plain(value):
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value

        Path(path).write_text(yaml.safe_dump(plain(asdict(self)), sort_keys=False))

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a mapping")
        init_names = {f.name for f in fields(cls) if f.init}
        derived_names = {f.name for f in fields(cls) if not f.init}
        unknown = data.keys() - init_names - derived_names
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in init_names})

    @classmethod
    def from_yaml(cls, path):
        return cls.from_dict(load_yaml_mapping(path))

    def assert_model_compatible(self, other):
        """Reject architecture and input geometry changes when loading weights."""
        from .model.pillar_ops import vfe_feature_opts

        names = ("model_variant", "vfe_type", "cell_size", "x_min", "x_max",
                 "y_min", "y_max", "z_min", "z_max", "col_axis", "row_axis",
                 "flip_cols", "flip_rows", "max_points_per_voxel", "height_clip")
        differences = [name for name in names if getattr(self, name) != getattr(other, name)]
        if self.model_variant != "heightmap" and other.model_variant != "heightmap":
            for name in ("scatter_config", "backbone_2d_config"):
                if getattr(self, name) != getattr(other, name):
                    differences.append(name)
            for name in ("NUM_FILTERS", "USE_NORM"):
                if self.vfe_config.get(name) != other.vfe_config.get(name):
                    differences.append(f"vfe_config.{name}")
            if vfe_feature_opts(self.vfe_config) != vfe_feature_opts(other.vfe_config):
                differences.append("vfe_config.feature_options")
        if differences:
            raise ValueError("Configuration does not match checkpoint model/preprocessing: "
                             + ", ".join(differences))
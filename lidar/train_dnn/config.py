"""Central configuration for the BEV occupancy training pipeline."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


class EasyDict(dict):
    """Dict that also supports attribute access, used by the model_cfg blocks."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


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
    cell_size: float = 0.1          # 0.1 m x 0.1 m = 0.01 m^2 per cell
    # z spans the full z range so every point falls into a single z pillar bucket
    pillar_dims: list[float] = field(default_factory=lambda: [0.1, 0.1, 20.0])
    # Which cloud axis feeds the grid column (j) and row (i). 0=x, 1=y, 2=z
    col_axis: int = 0
    row_axis: int = 1
    flip_rows: bool = False
    flip_cols: bool = False
    num_point_features = 3

    # ---- Point sampling ----
    #max_points: int = 60000
    #min_points: int = 16

    # ---- Optimization ----
    epochs: int = 50
    batch_size: int = 4
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

    # ---- Derived ----
    grid_h: int = field(init=False, default=0)
    grid_w: int = field(init=False, default=0)
    grid_size: list[int] = field(init=False, default_factory=lambda: [0, 0])

    vfe_config: dict = field(default_factory=lambda: EasyDict({
        "WITH_DISTANCE": False,
        "USE_ABSLOTE_XYZ": True,
        "USE_NORM": True,
        "NUM_FILTERS": [ 64, 64 ]}))

    scatter_config: dict = field(default_factory=lambda: EasyDict({
        "NUM_BEV_FEATURES": 64}))

    backbone_2d_config: dict = field(default_factory=lambda: EasyDict({
        "LAYER_NUMS": [3, 5, 5],
        "LAYER_STRIDES": [2, 2, 2],
        "NUM_FILTERS": [64, 128, 256],
        "UPSAMPLE_STRIDES": [2, 4, 8],
        "NUM_UPSAMPLE_FILTERS": [128, 128, 128]}))

    def __post_init__(self):
        self.grid_w = int(round((self.x_max - self.x_min) / self.cell_size))
        self.grid_h = int(round((self.y_max - self.y_min) / self.cell_size))
        if self.grid_w <= 0 or self.grid_h <= 0:
            raise ValueError("Invalid BEV extent / cell_size")
        if self.col_axis == self.row_axis:
            raise ValueError("col_axis and row_axis must differ")

        self.grid_size = [self.grid_w, self.grid_h, 1]

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))
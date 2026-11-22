# LiDAR Segmentation Dataset and Model

This repository generates simulated LiDAR data and trains a BEV occupancy model. The recommended model is the **lite pillar model**: it keeps a learned point-to-pillar encoder and uses a compact 2D BEV network. It is the main training path and the supported path for exporting a Torch-free ONNX inference bundle.

## Workflow

1. Generate Blender scenes. From the repository root, run:

   ```bash
   cd lidar
   blenderproc run lidar_generate_dataset.py
   cd ..
   ```

   The scene-generation script uses the source Blender scene and its dataset configuration. The source scene is not provided publicly at this time. For dataset configuration, check `lidar/dataset_config.py` and the configured paths before generating the scenes.

2. Export the generated scenes for Gazebo. Update the `SCENES_DIR` in `gazebo/export_all.sh` if necessary and run:

   ```bash
   cd gazebo
   ./export_all.sh
   cd ..
   ```

   After this, make sure Gazebo can find the generated models and worlds through `GZ_SIM_RESOURCE_PATH` environment variable.

3. Run the simulations. Check the `SCENES_DIR` variable in `gazebo/runsimall.sh` first; it must point to the generated Blender scenes. Then run:

   ```bash
   cd gazebo
   ./runsimall.sh
   cd ..
   ```

   This produces the simulation recordings and LiDAR dataset used for training the DNN. Set `data_root` in the training YAML to the directory containing the generated PLY files.

## Train the Lite Model

Configuration files are in [`lidar/configs/`](./lidar/configs/). Install
PyYAML in the Python environments used for training, export, TIDL compilation
and ONNX/ROS inference:

```bash
python -m pip install 'PyYAML>=6.0'
```

Edit `data_root` and `out_dir` in the YAML, then run training from the
`lidar` directory:

```bash
cd lidar
python -m gnd_seg_dnn.train --config configs/lite_static_04.yaml
```

Edit or copy the YAML files to configure a model. The supplied
[`lite_static_04.yaml`](./lidar/configs/lite_static_04.yaml) describes the
0.4 m static baseline; [`single_z16.yaml`](./lidar/configs/single_z16.yaml)
describes the single-layer search candidate. Paths in YAML are resolved
relative to the working directory. Unspecified fields use the defaults in
`Config`; derived geometry is recomputed from the extents and `cell_size`.
Training settings are read from YAML. Defaults in the supplied files include
100 epochs, batch size 8, a 2-epoch warmup followed by cosine decay,
and early stopping after 10 epochs without an IoU improvement above `0.001`.
Set `save_visualizations: false` or `point_stats: false` to skip those
outputs. To resume, add `--resume /path/to/checkpoint.pt`.

Evaluate with the saved run configuration and checkpoint:

```bash
python -m gnd_seg_dnn.evaluate --run-dir lite_static_04_yaml --split val
```

Evaluation reads `config.yaml` and `best.pt` from the run directory and checks
that model and input geometry match. Evaluation options such as `--data`,
`--threshold`, `--split`, `--workers`, and `--batch-size` can be overridden
on the command line. After export, add `--backend onnx` to evaluate the ONNX
pipeline or `--backend tidl` to evaluate compiled TIDL artifacts; use
`--pfn-on-cpu` for a TIDL build without the PFN.

### Cell size and pillar encoder

The BEV cell size is also the pillar size, the local-map resolution and the global-map resolution; they cannot differ. Valid sizes divide the 12.8 m x 51.2 m grid into whole cells and give a grid whose sides are divisible by 8 (for example 0.1, 0.2, 0.4, 0.8 or 1.6 m); other values are rejected with an error.

Set `cell_size: 0.4` in the YAML for 0.4 m pillars.

Coverage dilation (which cells count as observed) and the hole fill used to build training labels have a fixed reach of about 1.6 m, so their kernel in cells scales automatically with the cell size (7 cells at 0.2 m, 3 at 0.4 m, none at 0.8 m). Retrain when changing the cell size or this behaviour.

Two pillar encoders are available:

- `vfe_type: dynamic`: uses every point in a pillar.
- `vfe_type: static`: keeps at most `max_points_per_voxel` points per pillar, dropping later ones. This is the variant whose pillar network can be compiled by TIDL; see below.

The separate `model_variant: statpillars` encoder computes 15 statistics per
occupied pillar: minimum, linearly interpolated 25th percentile, mean, 75th
percentile and maximum for each of x, y and z, relative to the pillar center
in meters. The z center is the midpoint of the configured z bounds, as in the
pillar VFE. A multithreaded C++ voxelizer counts all points per pillar but
stores at most `max_points_per_voxel` (32 in the supplied config); statistics
are computed from the stored points. When a pillar overflows, atomic slot
assignment can select different points depending on thread scheduling. The
encoder produces a 15-channel statistics grid. With `statpillars_projection:
true` (the default), a bias-free 1x1 convolution at the start of the backbone
projects it to 32 channels; empty cells remain zero without a mask. Set the
option to `false` to feed 15 channels directly to the backbone instead.
Points stay on CPU even during GPU training; only the statistics grid is
transferred to the model device.

Build the C++ extension on each machine with
`cd lidar && python setup.py build_ext --inplace`. Rebuild existing extensions
after updating. Existing StatPillars checkpoints with the projection in the
encoder are incompatible and must be retrained; re-export ONNX bundles afterward.
`voxelizer_threads` controls the CPU thread count.

To train and evaluate this model from the `lidar` directory:

```bash
python -m gnd_seg_dnn.train --config configs/statpillars_xyz_stats_04.yaml
python -m gnd_seg_dnn.evaluate --run-dir statpillars_xyz_stats_04 --split test
```

Update `data_root` and `test_fraction` in the YAML for your dataset; the
test split is empty when `test_fraction` is zero or the scene count is too
small. If the output directory already exists, training appends a
number to the new run directory; pass that directory to evaluation.

```bash
python -m gnd_seg_dnn.train --config configs/lite_static_04.yaml
```

A lighter point encoder uses fewer input features and narrower layers. In
`vfe_config`, `ABS_POINT_COORDS: [2]` feeds only absolute z and
`REL_POINT_COORDS: [0, 1]` selects x/y offsets from the pillar's geometric
center. Both lists use indices 0=x, 1=y, 2=z (empty lists are allowed);
features are emitted in x/y/z order regardless of list order. All three
offsets from the mean of retained points remain included. Set
`WITH_DISTANCE_FROM_PC_ORIG: true` to add distance from the coordinate origin.
`NUM_FILTERS` sets the PFN widths; `scatter_config.NUM_BEV_FEATURES` must equal
the final width. The coordinate options apply to static and dynamic VFEs.
Changing features needs retraining, and the C++ voxelizer must be rebuilt
(`python setup.py build_ext --inplace`) on every machine running the static
inference pipeline:

```bash
python -m gnd_seg_dnn.train --config configs/single_z16.yaml
```

The old names `WITH_DISTANCE_TO_PC_ORIG`, `USE_ABSLOTE_XYZ`, `ABS_AXES`
and `USE_CENTER_Z` are no longer accepted, including in embedded checkpoint
configs. Existing YAMLs and checkpoint configurations must be manually
migrated before use: rename the distance field, replace the absolute settings
with `ABS_POINT_COORDS`, and replace `USE_CENTER_Z` with `[0, 1, 2]` when
true or `[0, 1]` when false. If `ABS_AXES` was present, use its value;
otherwise use `[0, 1, 2]` for `USE_ABSLOTE_XYZ: true` or `[]` for false.
Re-export ONNX/TIDL bundles after migration; only modify checkpoints from
trusted sources, keeping backups.

Static pillar training uses the PyTorch voxelizer; the static C++ voxelizer is
used in eval mode on CPU tensors. StatPillars uses its C++ voxelizer for both
training and inference.

### Static PFN architecture search

The search compares the saved `lite_static_04/best.pt` checkpoint and a
freshly trained baseline against a small, explicit set of cheaper static PFNs.
It varies the point cap (32, 16, or 8), PFN depth/width and input features,
while keeping the pillar size at **0.4 m**, the final PFN width at 32 and the
BEV network unchanged. Every architecture is trained from scratch with the
same settings, scene split and seeds as the retrained baseline.

From `lidar/`, using the same Python environment as training:

```bash
# Inspect candidates and projection MACs without training.
python -m gnd_seg_dnn.search_pfn --out pfn_search --dry-run

# Full search: by default nine candidates plus a retrained baseline, each
# using the checkpoint's epoch budget and early-stopping settings.
python -m gnd_seg_dnn.search_pfn --baseline lite_static_04/best.pt \
    --data /path/to/dnn_dataset --out pfn_search
```

`--out` must be a new directory; existing results are never overwritten.
To limit a preliminary run, use for example
`--epochs 10 --candidates narrow16 single16 z16 --workers 4`. Short runs are
only screening runs: repeat promising candidates in a **new** output directory
using the full baseline budget and, preferably, several seeds, e.g.
`--candidates narrow16 single16 z16 --seeds 0 1 2`. Each seed also retrains
the baseline. `--batch-size` can override the batch size for all trials.
For a fair comparison, use the same dataset contents and split as the original
run; `splits.json` records the file paths used, but does not detect changed
file contents. The saved checkpoint is re-evaluated on that validation split.

The search writes each trial's config, history and `best.pt`, plus
`baseline_evaluation.json` and `results.json`. Candidates are ranked by mean
best validation IoU over the requested seeds. A `winner` is reported only if
its mean beats both the retrained baseline mean and the re-evaluated saved
checkpoint; there may be no winner. These are validation-set comparisons, not a
guarantee of improvement on new scenes. Confirm finalists on a separate,
previously untouched test set and consider repeated seeds before deployment.
The best checkpoint for each individual seed is in its trial directory.

The compute figure is the sum of PFN linear/1x1 convolution multiply-accumulates
for one fully padded exported frame (`grid_h * grid_w` pillars), not total
model compute or measured latency. It excludes voxelization, feature
decoration, normalization, pooling, scatter and the BEV network. Reducing the
point cap also drops points and may change accuracy. Verify actual latency and
TIDL compatibility on the target hardware after exporting the chosen trial
with the normal static export procedure below.

If the output directory already exists, training chooses a numbered sibling such as `dnn_runs_lite_001`. Each run contains:

- `best.pt` and `last.pt` checkpoints. The log reports the epoch and validation IoU whenever a new best checkpoint is saved.
- `history.json` with per-epoch train and validation metrics.
- `splits.txt` listing the training and validation scenes and frame counts.
- `config.yaml`, an exact copy of the input YAML, periodic epoch checkpoints,
  and validation visualizations for best-IoU epochs.

The scene split is grouped by scene, so frames from one scene are not divided between training and validation. The split seed and scene lists are recorded with the run.

## Export for ONNX Runtime

Export the best checkpoint from the run directory. Use the actual numbered output directory if training selected one:

```bash
python -m gnd_seg_dnn.export_onnx --run-dir lite_static_04_yaml
```

The export directory `lite_static_04_yaml/onnx/` contains:

- `bev_net.onnx`: the dense BEV network.
- `pillar_vfe.npz`: the lite point encoder's folded weights and geometry.
- `config.yaml`: model variant and BEV geometry for inference.

The ONNX file contains the dense network only. The point encoder runs in NumPy before ONNX Runtime. Exporting requires the training environment with PyTorch and ONNX installed; the inference process does not need PyTorch.

## Prepare the LiDAR environment

From the `lidar` directory, source the setup script. It exports `DNN_ROOT` and the LiDAR package path, then builds the C++ voxelizer extension in place:

```bash
cd lidar
source ./setup_env.sh
```

The setup uses the active `python` executable and requires `pybind11` plus a C++17 compiler. Activate the intended virtual environment before sourcing it. Re-source the script after changing environments.

## Export a Static-Pillar Model (C++ voxelizer + PFN ONNX + BEV ONNX)

A checkpoint trained with `vfe_type: static` exports as a deployable pipeline:

```
points -> voxelize (C++) -> decorate (C++) -> PFN (ONNX) -> scatter (NumPy) -> BEV net (ONNX)
```

1. Build the C++ extension once on every machine that runs the pipeline (needs a C++17 compiler and `pybind11`):

   ```bash
   cd lidar
   pip install pybind11
   python setup.py build_ext --inplace
   ```

2. Export, choosing a fixed maximum pillar count for the PFN (TIDL needs static shapes):

   ```bash
   python -m gnd_seg_dnn.export_onnx --run-dir lite_static_04_yaml \
     --max-pillars 4096
   ```

   The export directory contains `pfn.onnx` (input `1 x F x max_pillars x max_points`, using only Conv, Relu, MaxPool, Resize and Concat), `bev_net.onnx` and `config.yaml`. The script ends with a check of the whole pipeline against the PyTorch model and prints the pillar count used and the maximum logit difference.

3. Choose `--max-pillars` from real data at your cell size. The graph always processes the full padded size, so too large wastes compute; too small drops real pillars (the node logs a warning and the check shows a larger logit difference). `config.yaml` records the value.

Run the node using the run directory:

```bash
python -m gnd_seg_dnn.ros2_infer --run-dir lite_static_04_yaml --backend onnx
```

The voxelizer and decoration use `voxelizer_threads` from the YAML. Compile
TIDL models using the TI toolchain:

```bash
python -m gnd_seg_dnn.tidl_compile --run-dir lite_static_04_yaml \
  --calib-dir /path/to/calibration/clouds
python -m gnd_seg_dnn.ros2_infer --run-dir lite_static_04_yaml --backend tidl
```

Artifacts are stored in `onnx/tidl_artifacts/`. For a PFN compiled with
`--skip-pfn`, add `--pfn-on-cpu` to the TIDL inference command. TIDL requires
the provider and corresponding artifacts; it does not silently select the
CPU backend if the provider is unavailable. Use `--no-timings` to disable
timing logs.

## Run ROS 2 Inference Without PyTorch

Install NumPy and ONNX Runtime in the inference environment, and source the ROS 2 and workspace setup files. StatPillars uses the C++ voxelizer extension to skip non-finite points and reuses pillar counts for coverage when the coverage axes are X and Y; other configurations use a separate coverage pass. Build the extension for StatPillars or static pillars on the target with `python setup.py build_ext --inplace` from `lidar`.

From `lidar`, launch with the exported model:

```bash
cd lidar
python -m gnd_seg_dnn.ros2_infer --run-dir lite_static_04_yaml --backend onnx
```

The run directory holds `config.yaml`, `best.pt`, and the `onnx/` directory.
The latter contains export-specific `config.yaml` (including the fixed pillar
count), the ONNX files, and optionally `tidl_artifacts/`. ONNX CPU inference
uses `CPUExecutionProvider`; select `--backend tidl` for TIDL. Old bundles
must be re-exported into `onnx/`, then recompiled for TIDL if needed.
Export refuses to overwrite a nonempty `onnx/` directory, and compilation
refuses to overwrite nonempty artifact directories; move or remove the old
outputs deliberately before regenerating them.

The node retains a PyTorch backend for desktop use:

```bash
python -m gnd_seg_dnn.ros2_infer --run-dir lite_static_04_yaml --backend torch
```

Use `--device cpu` or `--device cuda` for Torch; `auto` is the default.
`--world-frame` sets the map frame and `--threshold` sets the occupancy
threshold. `--voxelizer-threads` overrides the static or StatPillars CPU voxelizer count;
`--amp` / `--no-amp` control Torch CUDA mixed precision. Pass ROS options
after `--ros-args`. The Torch backend requires
PyTorch. The ONNX backend loads no PyTorch modules.

## SemanticKITTI Ground Segmentation

> **Scope.** This section trains a **binary ground / non-ground** segmenter and
> reports it with the metric defined by the GndNet and KAIST
> ground-segmentation benchmarks. That is **not one of the six official
> SemanticKITTI benchmarks** ([tasks](https://semantic-kitti.org/tasks.html):
> semantic segmentation, panoptic, 4D panoptic, moving-object segmentation,
> semantic scene completion, open-world instance segmentation), and the numbers
> here are **not comparable** with the SemanticKITTI leaderboards.
>
> In particular the `mIoU` reported below averages only **C = 2** classes
> (ground, non-ground) and is close to 87. The official semantic-segmentation
> mIoU averages **19-20 classes** over per-point labels and sits around 60-70
> on published leaderboard entries. When quoting work from this repository,
> cite the **ground-class IoU** (about 85.7 at 0.4 m) and name it as GndNet/
> KAIST-protocol binary ground segmentation, never as a SemanticKITTI mIoU.
>
> Producing an official SemanticKITTI result would need a 19-class point-wise
> head (this model emits one occupancy value per BEV cell), a different loss,
> and evaluation on the secrets test labels via CodaBench.

The same training stack runs on SemanticKITTI odometry scans, so results can be
compared with the ground-segmentation literature. Set `dataset: kitti` and
point `data_root` at the `sequences` directory (the one holding `00`, `01`, ...
each with `velodyne/` and `labels/`):

```yaml
data_root: /mnt/data/SemanticKITTI/raw/dataset/sequences
dataset: kitti
train_sequences: [0, 1, 2, 3, 4, 5, 6]
val_sequences: [7]
test_sequences: [8]
frame_stride: 1
label_fill_reach_m: 0.0        # no hole-fill dilation on real scans
kitti_ignore_classes: [0, 1, 70]   # unlabeled, outlier, vegetation
```

`train_sequences` accepts `7`, `"07"`, `"00-06"` or a list of those, and each
sequence stays wholly inside one split. `configs/kitti_statpillars_dseg_04.yaml` is
a ready 0.4 m StatPillars configuration: a 51.2 m x 51.2 m window (91-97% of a
scan), `z` in `[-8, 3]` m relative to the sensor, no `z` inversion.

The task stays binary BEV occupancy: a cell is positive when a scored
non-ground point lands in it, where ground is `road`, `parking`, `sidewalk`,
`other-ground`, `lane-marking` and `terrain` (`kitti.py`). Intensity is unused,
as in the simulator pipeline.

```bash
cd lidar
python -m gnd_seg_dnn.train --config configs/kitti_statpillars_dseg_04.yaml
python -m gnd_seg_dnn.evaluate_kitti --run-dir kitti_statpillars_dseg_04 --split test
```

`evaluate_kitti` converts the BEV prediction back to a label per point and
reports the protocol of the KAIST ground-segmentation benchmark (Patchwork,
Patchwork++, TRAVEL): point-wise Precision/Recall/F1 averaged per frame, ground
classes as above, `unlabeled`/`outlier` never scored and vegetation excluded
(`--with-vegetation` for the alternative variant). Options include
`--sequences 08 09`, `--max-frames`, `--threshold`, `--sweep`,
`--per-frame-out` and `--scope {window,scan}`. Use `scan` to score all labeled
points, where out-of-window points count as predicted non-ground.

Every evaluation also prints reference baselines on the same frames, because
these metrics saturate and a bare IoU says little on its own:

| Baseline | Meaning |
| --- | --- |
| all ground / all obstacle | trivial predictors |
| **height rule `REACH`** | classical segmentation: a point is ground when within `REACH` of its own cell's lowest point. Defaults 0.1 / 0.2 / 0.3 m; `--height-baselines` overrides, and passing no values disables it |
| cell oracle | perfect per-cell decision given this BEV grid, i.e. the quantisation ceiling |

The height rule matters because the label definition is close to *"is this
point within ~0.2 m of the local ground surface"*. On SemanticKITTI it reaches
~82-84 IoU with no training at all, so it is the baseline any learned model
must actually beat. A model that loses to it is not solving more than a height
threshold; a model that matches it at much higher precision is doing something
useful. Note that coarse cells starve the point encoder (`max_points_per_voxel`
caps how many points per pillar reach the network: ~32% of points survive at
1.0 m, ~94% at 0.4 m, ~100% at 0.2 m), which is why the 1.0 m config
underperforms the height rule while 0.4 m does not.

### Champion recipe and hyper-parameter search

`configs/kitti_statpillars_gndnet_04.yaml` is the tuned 0.4 m recipe. A
sequential search over loss and optimisation settings
(`python -m gnd_seg_dnn.search_kitti`) improved the ground-class IoU on the
full GndNet validation split:

| Recipe | Precision | Recall | gIoU |
| --- | --- | --- | --- |
| GndNet (paper) | 84.1 | 99.3 | 83.6 |
| height rule 0.2 m (no training) | 93.3 | 88.4 | 84.8 |
| `focal_alpha: 0.75` (previous) | 98.4 | 86.1 | 84.9 |
| **`focal_alpha: 0.5` (champion)** | 98.1 | 87.2 | **85.7** |
| `loss_type: bce` (tie) | 97.8 | 87.4 | 85.7 |
| cell oracle (quantisation ceiling) | 100.0 | 89.3 | 90.6 |

The gain comes from recall: rebalancing the focal loss recovers ground points
the old setting under-segmented, at no cost in precision. `focal_gamma: 1.0`,
`lr: 0.001` and `warmup_epochs: 2` all made it worse. `label_fill_reach_m`
below 0.9 m is a no-op at 0.4 m cells because the dilation kernel stays at 1.

The search writes one directory per trial plus `results.json` and `summary.txt`,
runs a single training session at a time, and stops at its `--time-budget`.
Champion artifacts live in `lidar/kitti_statpillars_gndnet_04/`.

Two families of KITTI configs exist, differing only in their sequence split:

| Config | Split | Purpose |
| --- | --- | --- |
| `kitti_statpillars_dseg_04.yaml` | train 00-06, val 07, test 08 | DSegKITTI ground-segmentation protocol |
| `kitti_statpillars_gndnet_{10,04,02}.yaml` | train 00,02,03,04,06,08,10, val 01,05,07,09 | GndNet's 7/4 split, so results can be compared with Paigwar et al. (IROS 2020) |

The `gndnet` suffix names the **evaluation protocol** (split, ground classes,
metric), never the network: every config in this repository trains this
repository's StatPillars model. Evaluate them with `--protocol gndnet`, which
reproduces GndNet's point-wise Precision/Recall/IoU, drops unlabeled/outlier
points, treats vegetation as non-ground and reports ground-class IoU.

To score a checkpoint trained on the simulator data without retraining, pass
its run directory; the loader keeps the checkpoint's own BEV geometry, and
`--sequences` selects the frames:

```bash
python -m gnd_seg_dnn.evaluate_kitti --run-dir statpillars_xyz_stats_04 \
  --data /mnt/data/SemanticKITTI/raw/dataset/sequences --sequences 08
```

Note that this evaluates a model trained on a different sensor geometry and
label distribution, so treat it as a transfer baseline rather than a result
comparable with the published tables.

## Alternative Models

- **StatPillars:** set `model_variant: statpillars` in a YAML config to train on
  32-channel BEV features projected from per-pillar relative XYZ statistics.

  Export it with the same `export_onnx` command. Its `pillar_vfe.npz` stores
  the pillar geometry and preprocessing parameters. The CPU statistics
  encoder produces 15-channel calibration and runtime inputs; the projection
  and BEV network are exported together and compiled for the C7x:

  ```bash
  python -m gnd_seg_dnn.export_onnx --run-dir statpillars_xyz_stats_04
  python -m gnd_seg_dnn.tidl_compile --run-dir statpillars_xyz_stats_04 \
    --calib-dir /path/to/calibration/clouds
  python -m gnd_seg_dnn.evaluate --run-dir statpillars_xyz_stats_04 \
    --backend tidl --split val
  ```

  Use the actual training run directory, and choose representative calibration
  clouds. The C++ voxelizer extension must be built in both the compilation and
  inference environments. `--skip-pfn`, `--pfn-on-cpu` and PFN deny-list
  options do not apply to StatPillars. TIDL compilation and execution require
  the TI toolchain and provider, and must be validated on the target device.

  The former `model_variant: heightmap` name is accepted with a warning in
  existing configurations. Older one-channel or 15-channel heightmap
  checkpoints, and statistics-encoder checkpoints with the projection in the
  encoder, are incompatible and must be retrained.

- **Full model:** set `model_variant: full` in YAML to train the original, larger model. The full model is not accepted by the embedded ONNX exporter.

Compare validation IoU and board-side latency before choosing a deployment model. Parameter count alone does not establish the achieved inference rate; profile both point preprocessing and ONNX inference with representative point clouds on the target hardware.

## Example end-to-end commands

After sourcing [lidar/setup_env.sh](./lidar/setup_env.sh), [lidar/testall.sh](./lidar/testall.sh) provides separate example commands for training, evaluation, static ONNX export, and ROS 2 inference:

```bash
cd lidar
source ./setup_env.sh
./testall.sh train configs/lite_static_04.yaml
./testall.sh eval lite_static_04_yaml
./testall.sh export lite_static_04_yaml
./testall.sh ros2 lite_static_04_yaml onnx
```

The commands run one stage at a time so that a numbered training output
directory can be selected explicitly for later stages.

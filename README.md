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

   This produces the simulation recordings and LiDAR dataset used for training the DNN. The default training data root is `/home/ubuntu/shared/dnn_dataset`; override it with `--data` if the generated PLY files are elsewhere.

## Train the Lite Model

Run training from the `lidar` directory:

```bash
cd lidar
python -m gnd_seg_dnn.train --lite --out dnn_runs_lite
```

The command uses the lite model explicitly. Defaults include 100 epochs, batch size 8, a 2-epoch learning-rate warmup followed by cosine decay, and early stopping after 10 epochs without a validation IoU improvement greater than `0.001`.

Useful overrides:

```bash
python -m gnd_seg_dnn.train --lite \
  --data /path/to/dnn_dataset \
  --out dnn_runs_lite \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.002
```

### Cell size and pillar encoder

The BEV cell size is also the pillar size, the local-map resolution and the global-map resolution; they cannot differ. Valid sizes divide the 12.8 m x 51.2 m grid into whole cells and give a grid whose sides are divisible by 8 (for example 0.1, 0.2, 0.4, 0.8 or 1.6 m); other values are rejected with an error.

```bash
python -m gnd_seg_dnn.train --lite --cell-size 0.4 --out dnn_runs_lite_04
```

Coverage dilation (which cells count as observed) and the hole fill used to build training labels have a fixed reach of about 1.6 m, so their kernel in cells scales automatically with the cell size (7 cells at 0.2 m, 3 at 0.4 m, none at 0.8 m). Retrain when changing the cell size or this behaviour.

Two pillar encoders are available:

- `--vfe dynamic` (default): uses every point in a pillar.
- `--vfe static`: keeps at most `--max-points` points per pillar, dropping later ones. This is the variant whose pillar network can be compiled by TIDL; see below.

```bash
python -m gnd_seg_dnn.train --lite --vfe static --cell-size 0.4 --out dnn_runs_static
```

Training always uses the PyTorch voxelizer. The C++ voxelizer described below is used only in eval mode on CPU tensors.

If the output directory already exists, training chooses a numbered sibling such as `dnn_runs_lite_001`. Each run contains:

- `best.pt` and `last.pt` checkpoints. The log reports the epoch and validation IoU whenever a new best checkpoint is saved.
- `history.json` with per-epoch train and validation metrics.
- `splits.txt` listing the training and validation scenes and frame counts.
- `config.json`, periodic epoch checkpoints, and validation visualizations for best-IoU epochs.

The scene split is grouped by scene, so frames from one scene are not divided between training and validation. The split seed and scene lists are recorded with the run.

## Export for ONNX Runtime

Export the best checkpoint from the run directory. Use the actual numbered output directory if training selected one:

```bash
python -m gnd_seg_dnn.export_onnx \
  --ckpt dnn_runs_lite/best.pt \
  --out exports/lite
```

The export directory contains:

- `bev_net.onnx`: the dense BEV network.
- `pillar_vfe.npz`: the lite point encoder's folded weights and geometry.
- `config.json`: model variant and BEV geometry for inference.

The ONNX file contains the dense network only. The point encoder runs in NumPy before ONNX Runtime. Exporting requires the training environment with PyTorch and ONNX installed; the inference process does not need PyTorch.

## Prepare the LiDAR environment

From the `lidar` directory, source the setup script. It exports `DNN_ROOT` and the LiDAR package path, then builds the C++ voxelizer extension in place:

```bash
cd lidar
source ./setup_env.sh
```

The setup uses the active `python` executable and requires `pybind11` plus a C++17 compiler. Activate the intended virtual environment before sourcing it. Re-source the script after changing environments.

## Export a Static-Pillar Model (C++ voxelizer + PFN ONNX + BEV ONNX)

A checkpoint trained with `--vfe static` exports as a deployable pipeline:

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
   python -m gnd_seg_dnn.export_onnx \
     --ckpt dnn_runs_static/best.pt \
     --out exports/static \
     --max-pillars 4096
   ```

   The export directory contains `pfn.onnx` (input `1 x F x max_pillars x max_points`, using only Conv, Relu, MaxPool, Resize and Concat), `bev_net.onnx` and `config.json`. The script ends with a check of the whole pipeline against the PyTorch model and prints the pillar count used and the maximum logit difference.

3. Choose `--max-pillars` from real data at your cell size. The graph always processes the full padded size, so too large wastes compute; too small drops real pillars (the node logs a warning and the check shows a larger logit difference). `config.json` records the value.

Run the node exactly as for the lite export; it detects `"vfe_type": "static"` in `config.json` and loads `pfn.onnx` from the same directory (override with `DNN_PFN`):

```bash
DNN_BACKEND=onnx \
DNN_ONNX_MODEL=/path/to/exports/static/bev_net.onnx \
python -m gnd_seg_dnn.ros2_infer
```

The voxelizer and decoration use 4 threads by default (`voxelizer_threads` in `config.py`). TIDL compilation of `pfn.onnx` and `bev_net.onnx` is done with your own TI toolchain; the node itself runs both models with ONNX Runtime's CPU provider.

The node logs pillar encoding and network timings separately by default; set `self.log_exec_time = False` in `ros2_infer.py` to disable the extra timing output. The callback log reports `pillar-enc` and `net` times.

## Run ROS 2 Inference Without PyTorch

Install NumPy and ONNX Runtime in the inference environment, and source the ROS 2 and workspace setup files. From `lidar`, launch with the exported model:

```bash
cd lidar
DNN_BACKEND=onnx \
DNN_ONNX_MODEL=/path/to/exports/lite/bev_net.onnx \
python -m gnd_seg_dnn.ros2_infer
```

By default, the node reads `config.json` and `pillar_vfe.npz` from the ONNX model's directory. Set `DNN_CONFIG` or `DNN_ENCODER` to override either path. This backend uses ONNX Runtime's `CPUExecutionProvider`; it runs on the CPU and does not currently offload to GPU.

The node retains a PyTorch backend for desktop use:

```bash
DNN_BACKEND=torch \
DNN_WEIGHTS=/path/to/dnn_runs_lite/best.pt \
python -m gnd_seg_dnn.ros2_infer
```

The Torch backend requires PyTorch. The ONNX backend loads no PyTorch modules.

## Alternative Models

- **Heightmap:** train a one-channel BEV input containing normalized per-cell maximum height:

  ```bash
  python -m gnd_seg_dnn.train --heightmap --out dnn_runs_heightmap
  ```

  Export it with the same `export_onnx` command. Its `pillar_vfe.npz` stores heightmap geometry and clipping settings rather than learned point-encoder weights.

- **Full model:** omit `--lite` and `--heightmap` to train the original, larger model. The full model is not accepted by the embedded ONNX exporter.

Compare validation IoU and board-side latency before choosing a deployment model. Parameter count alone does not establish the achieved inference rate; profile both point preprocessing and ONNX inference with representative point clouds on the target hardware.

## Example end-to-end commands

After sourcing [lidar/setup_env.sh](./lidar/setup_env.sh), [lidar/testall.sh](./lidar/testall.sh) provides separate example commands for training, evaluation, static ONNX export, and ROS 2 inference:

```bash
cd lidar
source ./setup_env.sh
./testall.sh train
RUN_DIR=./dnn_runs_static ./testall.sh eval
RUN_DIR=./dnn_runs_static ./testall.sh export
ONNX_MODEL=./exports/static/bev_net.onnx ./testall.sh ros2
```

Set `DATA_ROOT`, `RUN_DIR`, `EXPORT_DIR`, `CKPT`, and `ONNX_MODEL` to match the local dataset and generated artifacts. The commands intentionally run one stage at a time so that a numbered training output directory can be selected explicitly for later stages.

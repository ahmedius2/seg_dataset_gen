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

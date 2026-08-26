# Synthetic Aerial Segmentation Dataset Generator

Generates a three-class instance segmentation dataset (traversable, obstacle,
and target)
by rendering your Blender scene from simulated fixed-wing drone viewpoints,
then converts the outputs into **Ultralytics YOLO11-seg** format.

---

## File overview

| File | Purpose |
|---|---|
| `generate_dataset.py` | BlenderProc script — renders RGB images + class and instance maps |
| `convert_to_yolo.py` | Standard Python script — converts maps → YOLO-seg labels |
| `README_dataset_gen.md` | This file |

---

## Step 0 — Install dependencies

```bash
# BlenderProc (installs its own Blender internally)
pip install blenderproc

# For convert_to_yolo.py (run with your system / venv Python, NOT blenderproc run)
pip install opencv-python numpy pyyaml imageio

# For training
pip install ultralytics
```

---

## Step 1 — Prepare your Blender scene

Before running the generator, do the following inside Blender:

### 1a. Find exact object names
Select each object → look at **Properties panel → Object Properties (orange square icon) → Name**.

You need:
- The **exact name** of the ground plane mesh (default assumption: `"Ground"`)
- The names of obstacle meshes (or leave `OBSTACLE_KEYWORDS = []` to auto-detect everything else as obstacle)

### 1b. Make sure the ground plane covers the full 300×300 m area
The camera will sometimes be positioned at the edges of the 50×50 m inner zone; the ground plane must extend far enough not to show any void.

### 1c. Remove any existing camera from the scene (optional but recommended)
BlenderProc creates its own camera. Having two cameras in the scene will not break anything, but it is cleaner to remove the original one.

### 1d. UV-unwrap the ground plane
The texture randomisation works through UV coordinates. Select the ground plane → **Tab (Edit Mode)** → **U → Smart UV Project** → **OK**.  
Without UV coordinates, the textures will not appear on the ground.

---

## Step 2 — Edit the CONFIG block in `generate_dataset.py`

Open `generate_dataset.py` and update the values at the top:

```python
BLEND_FILE      = "/absolute/path/to/your/scene.blend"
TEXTURES_ROOT   = "/absolute/path/to/ground_textures"
OUTPUT_DIR      = "/absolute/path/to/output"
NUM_IMAGES      = 100
GROUND_OBJ_NAME = "Ground"          # exact name from Step 1a
TARGET_OBJ_NAME = "Target"          # exact target object name
```

Also check the drone camera parameters and adjust to your real drone specs:

```python
DRONE_ALT_MIN  = 35.0   # metres above ground
DRONE_ALT_MAX  = 60.0
CAM_HFOV_DEG   = 70.0   # horizontal FOV of the camera
MAX_TILT_DEG   = 8.0    # max off-nadir tilt
```

And make sure `INNER_CX`, `INNER_CY`, `INNER_HALF` match the actual world-space
coordinates of your 50×50 m obstacle area.

---

## Step 3 — Generate images + masks

```bash
blenderproc run generate_dataset.py
```

This will print progress and write to `OUTPUT_DIR/`:
```
output/
├── images/
│   ├── 0000.png   ← RGB render (640×640)
│   ├── 0001.png
│   └── ...
├── masks/
│   ├── 0000.png   ← class map (0=traversable, 1=obstacle, 2=target)
│   ├── 0001.png
│   └── ...
├── instances/
│   ├── 0000.png   ← 16-bit BlenderProc instance IDs
│   └── ...
├── preview/
│   ├── 0000_classes.png   ← colorized classes for inspection
│   ├── 0000_instances.png ← colorized instance IDs
│   └── 0000_overlay.png   ← RGB with class colors overlaid
└── dataset_meta.json
```

> **Tip — check a few pairs visually before generating 100 images.**
> Compare the RGB image with the class map and instance map. Every visible
> object should have a non-zero instance ID and the `Target` object should
> have class ID 2.

---

## Step 4 — Edit the CONFIG block in `convert_to_yolo.py`

```python
OUTPUT_DIR = "/absolute/path/to/output"      # same as above
YOLO_DIR   = "/absolute/path/to/yolo_dataset"
```

---

## Step 5 — Convert to YOLO-seg format

Run this with your **system / venv Python** (not `blenderproc run`):

```bash
python convert_to_yolo.py
```

Output structure:
```
yolo_dataset/
├── images/
│   ├── train/   (80 %)
│   └── val/     (20 %)
├── labels/
│   ├── train/   (.txt files with polygon annotations)
│   └── val/
└── dataset.yaml
```

Enable `DEBUG = True` in `convert_to_yolo.py` to generate overlay images
showing the detected polygons drawn on top of the RGB images — useful for
verifying that the contour extraction is working correctly.

---

## Step 6 — Train YOLO11-seg

```bash
yolo segment train \
    data=/absolute/path/to/yolo_dataset/dataset.yaml \
    model=yolo11n-seg.pt \
    epochs=100 \
    imgsz=640 \
    batch=16
```

Start with `yolo11n-seg.pt` (nano) for fast iteration.
Switch to `yolo11s-seg.pt` or `yolo11m-seg.pt` once the pipeline is validated.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| All class-map pixels are 0 | No category IDs assigned | Check object names and the generator log |
| Target is labelled as obstacle | Target name mismatch | Set `TARGET_OBJ_NAME` to the exact Blender object name |
| Textures not showing | Ground has no UV map | UV-unwrap the ground plane in Blender (Step 1d) |
| `Object 'Ground' not found` | Name mismatch | Check the exact object name in Blender Properties panel |
| Renders are very dark | Sun not created / low energy | Check `setup_or_randomise_sun()` or increase `DRONE_ALT_MAX` |
| Empty YOLO label files | No obstacles visible in those views | Normal for some camera positions; these act as negative samples |
| Camera sees ground edge (void) | Ground plane too small | Extend the ground plane beyond 300×300 m in Blender |

---

## What to add next

- **Obstacle position randomisation** — in the generation loop, after loading the scene, translate/rotate obstacle objects to random positions within the inner area before each render.
- **HDRI sky randomisation** — replace the sun with random HDRI environment maps for better lighting variety.
- **Camera height variation** — already implemented; widen the altitude range for more scale variation.
- **More texture sets** — download additional Poly Haven ground/gravel/soil textures and drop them into `TEXTURES_ROOT`.

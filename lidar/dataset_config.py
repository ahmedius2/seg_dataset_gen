# Script usage: blenderproc run lidar_generate_dataset.py <output_dir> <num_scenes>

import sys
import os

OUT_DIR = "output"
NUM_SCENES = 10
if len(sys.argv) >= 2:
    OUT_DIR = sys.argv[1]
if len(sys.argv) >= 3:
    NUM_SCENES = int(sys.argv[2])

os.makedirs(OUT_DIR, exist_ok=True)

# Number of LiDAR frames (flight poses) to simulate per scene. The lawn-mower
# flight path is resampled to exactly this many evenly spaced poses.
NUM_FRAMES = 5

# Number of Blender frames used to let active obstacles settle on the passive
# ground before the cameras and LiDAR capture the scene.
PHYSICS_SETTLE_FRAMES = 120

# Export the first N generated scenes as .blend files (into each scene's
# output dir) for manual inspection in Blender. Set to 0 to disable.
NUM_SCENES_TO_EXPORT_BLEND = NUM_SCENES

MIN_DIST_BTW_START_TARGET = 50.0  # meters: start/target must be far apart to avoid trivial paths
PLANE_SIZE = 300.0               # visible ground plane size in meters
AREA_SIZE_M = 50.0               # meters: inner active region for obstacle generation and flight path
CELL_SIZE_M = 0.5                # 0.5x0.5 m grid cells
CELL_COUNT = int(AREA_SIZE_M / CELL_SIZE_M)
GRID_N = int(AREA_SIZE_M / CELL_SIZE_M)

# NOTE: increase amplitude and scale to make the terrain more bumpy, or decrease to make it flatter.

# Ground plane surface roughness: a subdivided grid perturbed with low-frequency
# Perlin noise (macro undulation) plus small per-vertex jitter (micro roughness),
# so the base terrain is no longer perfectly flat.
GROUND_NOISE_CELL_SIZE_M = 0.2    # meters per grid subdivision (smaller = more detail, more geometry)
GROUND_NOISE_AMPLITUDE_M = 0.05    # meters, max height of the low-frequency undulation
GROUND_NOISE_SCALE = 0.1         # Perlin noise frequency (per meter); smaller = broader bumps
GROUND_NOISE_MICRO_STD_M = 0.05   # meters, Gaussian per-vertex jitter (fine-grained roughness)

FLIGHT_ALT_MIN = 30.0
FLIGHT_ALT_MAX = 40.0

# Attitude jitter to simulate imperfect nadir pointing (deg), applied per-pose
MAX_PITCH_DEV_DEG = 8.0     # forward/backward tilt from vertical
MAX_ROLL_DEV_DEG = 8.0      # side-to-side tilt from vertical
MAX_YAW_JITTER_DEG = 5.0    # small heading wander (usually irrelevant for nadir FOV symmetry, kept for realism)

# --- Sensor presets ---
# RoboSense E1R: fully solid-state digital LiDAR (SPAD-SoC + 2D VCSEL, no
# moving parts), FOV 120(H) x 90(V) deg, angular resolution 0.625(H) x
# 0.625(V) deg -> resX = 120/0.625 = 192, resY = 90/0.625 = 144.
# Source: robosense.ai/en/IncrementalComponents/E1R
ROBOSENSE_E1R_FOV_X_DEG = 120.0
ROBOSENSE_E1R_FOV_Y_DEG = 90.0
ROBOSENSE_E1R_PARAMS = dict(
    resolutionX=192,
    fovX=ROBOSENSE_E1R_FOV_X_DEG,
    resolutionY=144,
    fovY=ROBOSENSE_E1R_FOV_Y_DEG,
)

# RoboSense EMX: automotive-grade digital LiDAR, 192 beams, FOV 140(H) x
# 20(V) deg, global angular resolution 0.08(H) x 0.1(V) deg ->
# resX = 140/0.08 ~= 1750, resY = 20/0.1 = 200.
# Source: shop.leodrive.ai/robosense-emx-192-kanalli-otomotiv-sinifi-yuksek-performansli-dijital-lidar-sensoru
ROBOSENSE_EMX192_FOV_X_DEG = 140.0
ROBOSENSE_EMX192_FOV_Y_DEG = 20.0
ROBOSENSE_EMX192_PARAMS = dict(
    resolutionX=1750,
    fovX=ROBOSENSE_EMX192_FOV_X_DEG,
    resolutionY=200,
    fovY=ROBOSENSE_EMX192_FOV_Y_DEG,
)

# Active sensor used by the dataset generator. Change to switch presets.
LIDAR_SENSOR_NAME = "robosense_emx192"
LIDAR_SENSOR_PARAMS = ROBOSENSE_EMX192_PARAMS
LIDAR_FOV_X_DEG = ROBOSENSE_EMX192_FOV_X_DEG
LIDAR_FOV_Y_DEG = ROBOSENSE_EMX192_FOV_Y_DEG

# Keep the RGB render camera geometry consistent with the active LiDAR's FOV
# using Blender's explicit camera angle properties.
RENDER_RES_X = 1750
RENDER_RES_Y = 200

# ----
# Mask-driven scene reconstruction config
# ----
# 100x100 px bird's-eye masks describe the inner AREA_SIZE x AREA_SIZE region.
# Black blobs = rubble piles, white = clear path, red blobs = barricades.

MASK_DIR = "/home/dho/work/ileri_otonom/seg_dataset_gen/scene_masks"
MASK_PX = 100                     # mask is MASK_PX x MASK_PX pixels
PIX_SIZE = AREA_SIZE_M / MASK_PX  # meters per mask pixel (50/100 = 0.5 m/px)

# Classification thresholds on 0..1 normalized RGB.
BLACK_LEVEL = 0.35                # pixel is "rubble" if max(R,G,B) < BLACK_LEVEL
RED_MIN = 0.5                     # red barricade: R high ...
RED_OTHER_MAX = 0.35              # ... while G and B are low

# Rubble elevation. Per-blob peak height is drawn uniformly in this range.
RUBBLE_MAX_HEIGHT_MIN = 1.0        # meters (configurable)
RUBBLE_MAX_HEIGHT_MAX = 5.0        # meters (configurable)
RUBBLE_SURFACE_MIN_NOISE = 0.05        # meters, per-vertex Gaussian roughness of rubble
RUBBLE_SURFACE_MAX_NOISE = 0.30        # meters, per-vertex Gaussian roughness of rubble

RUBBLE_MIN_BLOB_PX = 3             # ignore tiny specks smaller than this many px

# Barricade geometry (oriented box aligned to each red blob's major axis).
BARRICADE_HEIGHT = 1               # meters tall
BARRICADE_MIN_BLOB_PX = 1          # ignore tiny red specks

# Start/target sampling in mask space.
MIN_DIST_BTW_START_TARGET_PX = MIN_DIST_BTW_START_TARGET / PIX_SIZE

SOURCE_SCENE_PATH = "/home/dho/work/ileri_otonom/seg_dataset_gen/bl_scenes/newscene/rubbles.blend"
# Script usage: blenderproc run lidar_generate_dataset.py <output_dir>
import os

# When True, only scene geometry is generated (ground, obstacles, markers,
# optional .blend export) and the LiDAR scan / rendering step is skipped.
SKIP_RENDER_AND_SCAN = True

# Export the first N generated scenes as .blend files (into each scene's
# output dir) for manual inspection in Blender. Set to 0 to disable.
# Set to -1 to export all scenes.
NUM_SCENES_TO_EXPORT_BLEND = -1

# Seed for Python's `random` module (and numpy) so that a given seed always
# reproduces the same masks, obstacle layouts, ground noise, and flight jitter.
# Override via `blenderproc run ... -- --seed=N`.
SEED = 42

OUT_DIR = "output"
NUM_SCENES_PER_MASK = 2
os.makedirs(OUT_DIR, exist_ok=True)

# Number of LiDAR frames (flight poses) to simulate per scene. The lawn-mower
# flight path is resampled to exactly this many evenly spaced poses.
NUM_FRAMES_PER_SCENE = 1

# Number of Blender frames used to let active obstacles settle on the passive
# ground before the cameras and LiDAR capture the scene.
# PHYSICS_SETTLE_FRAMES = 120


MIN_DIST_BTW_START_TARGET = 50.0  # meters: start/target must be far apart to avoid trivial paths
PLANE_SIZE = 200.0               # visible ground plane size in meters
AREA_SIZE_M = 50.0               # meters: inner active region for obstacle generation and flight path
CELL_SIZE_M = 0.5                # 0.5x0.5 m grid cells
GRID_N = int(AREA_SIZE_M / CELL_SIZE_M)

# NOTE: increase amplitude and scale to make the terrain more bumpy, or decrease to make it flatter.

# Ground plane surface roughness: a subdivided grid perturbed with low-frequency
# Perlin noise (macro undulation) plus small per-vertex jitter (micro roughness),
# so the base terrain is no longer perfectly flat.
# Cell size is randomly chosen per scene from this set (meters per grid subdivision).
GROUND_NOISE_CELL_SIZE_CHOICES = [0.5, 1.0, 2.0]

# Amplitude, Perlin noise scale, and micro-jitter std all share one value per
# scene, linearly interpolated across the full run from MIN to MAX (step size
# is 1 / total_scenes, so the last scene lands on MAX).
GROUND_NOISE_VALUE_MIN = 0.01     # meters
GROUND_NOISE_VALUE_MAX = 0.15     # meters


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
    resolutionX=int(ROBOSENSE_E1R_FOV_X_DEG/0.625),
    fovX=ROBOSENSE_E1R_FOV_X_DEG,
    resolutionY=int(ROBOSENSE_E1R_FOV_Y_DEG/0.625),
    fovY=ROBOSENSE_E1R_FOV_Y_DEG,
)

# RoboSense EMX: automotive-grade digital LiDAR, 192 beams, FOV 140(H) x
# 20(V) deg, global angular resolution 0.08(H) x 0.1(V) deg ->
# resX = 140/0.08 ~= 1750, resY = 20/0.1 = 200.
# Source: shop.leodrive.ai/robosense-emx-192-kanalli-otomotiv-sinifi-yuksek-performansli-dijital-lidar-sensoru
ROBOSENSE_EMX192_FOV_X_DEG = 140.0 # NOTE, ACTUAL VALUE is 140, to save time we use 50.0
ROBOSENSE_EMX192_FOV_Y_DEG = 20.0
ROBOSENSE_EMX192_PARAMS = dict(
    resolutionX=int(ROBOSENSE_EMX192_FOV_X_DEG/0.08),
    fovX=ROBOSENSE_EMX192_FOV_X_DEG,
    resolutionY=int(ROBOSENSE_EMX192_FOV_Y_DEG/0.1),
    fovY=ROBOSENSE_EMX192_FOV_Y_DEG,
)

# Active sensor used by the dataset generator. Change to switch presets.
LIDAR_SENSOR_NAME = "robosense_emx192"
LIDAR_SENSOR_PARAMS = ROBOSENSE_EMX192_PARAMS
LIDAR_FOV_X_DEG = ROBOSENSE_EMX192_FOV_X_DEG
LIDAR_FOV_Y_DEG = ROBOSENSE_EMX192_FOV_Y_DEG

# Keep the RGB render camera geometry consistent with the active LiDAR's FOV
# using Blender's explicit camera angle properties.
RENDER_RES_X = LIDAR_SENSOR_PARAMS['resolutionX']
RENDER_RES_Y = LIDAR_SENSOR_PARAMS['resolutionY']

# ----
# Mask-driven scene reconstruction config
# ----
# 100x100 px bird's-eye masks describe the inner AREA_SIZE x AREA_SIZE region.
# Black blobs = rubble piles, white = clear path, red blobs = barricades.

MASK_DIR = "/home/dho/work/ileri_otonom/seg_dataset_gen/scene_masks"
MASK_PX = 100                     # mask is MASK_PX x MASK_PX pixels
PIX_SIZE = AREA_SIZE_M / MASK_PX  # meters per mask pixel (50/100 = 0.5 m/px)

# Mask merging: combine every 4 masks into a 2x2, (2*MASK_PX)x(2*MASK_PX) tile,
# then downscale back to MASK_PX x MASK_PX so it drops into the pipeline unchanged.
MASK_MERGE_MODE = True            # when True, generate+use merged masks instead of the raw ones
MASK_MERGE_RANDOM_ORDER = False    # False = merge consecutive masks; True = shuffle first
MASK_MERGE_DIR = "/home/dho/work/ileri_otonom/seg_dataset_gen/scene_masks/merged"

# Classification thresholds on 0..1 normalized RGB.
BLACK_LEVEL = 0.35                # pixel is "rubble" if max(R,G,B) < BLACK_LEVEL
RED_MIN = 0.5                     # red barricade: R high ...
RED_OTHER_MAX = 0.35              # ... while G and B are low

# Rubble elevation. Per-blob peak height is drawn uniformly in this range.
RUBBLE_MAX_HEIGHT_MIN = 1.0        # meters (configurable)
RUBBLE_MAX_HEIGHT_MAX = 5.0        # meters (configurable)
RUBBLE_SURFACE_MIN_NOISE = 0.0        # meters, per-vertex Gaussian roughness of rubble
RUBBLE_SURFACE_MAX_NOISE = 0.5        # meters, per-vertex Gaussian roughness of rubble

# Probability that a rubble pile is a flat-topped elevated wall (vertical
# sides, no dome-like curvature) instead of the usual Gaussian-faded mound.
RUBBLE_WALL_PROBABILITY = 0.2

RUBBLE_MIN_BLOB_PX = 3             # ignore tiny specks smaller than this many px

# Small barricades scattered along each rubble pile's boundary, facing inward.
RUBBLE_EDGE_BARRICADES_ENABLED = False  # set False to disable placing these entirely
RUBBLE_EDGE_BARRICADE_MIN = 0      # min number placed per rubble pile
RUBBLE_EDGE_BARRICADE_MAX = 2      # max number placed per rubble pile
RUBBLE_EDGE_BARRICADE_OFFSET_M = 0.5  # meters, pushed outward from the pile's edge

# Barricade geometry (oriented box aligned to each red blob's major axis).
BARRICADE_HEIGHT = 1               # meters tall
BARRICADE_MIN_BLOB_PX = 1          # ignore tiny red specks

# Regardless of how many red (barricade) blobs a mask has, at least this
# fraction of them are cleared into an open path; the rest stay blocked.
# Which blobs are cleared is chosen randomly per scene.
CLEARED_BARRICADE_FRACTION = 0.33

# Start/target sampling in mask space.
MIN_DIST_BTW_START_TARGET_PX = MIN_DIST_BTW_START_TARGET / PIX_SIZE

SOURCE_SCENE_PATH = "/home/dho/work/ileri_otonom/seg_dataset_gen/lidar/rubbles.blend"

# ----
# Background scattering (Buildings/Cars/Trees/Humans/Animals/Other)
# ----
# Objects from these source collections are copied and spread with Poisson-disk
# style placement over the outer PLANE_SIZE area, keeping the inner
# AREA_SIZE_M x AREA_SIZE_M mask-driven region clear. Placement order matters:
# buildings first, then cars, then trees, humans, animals, and finally other.
# Density N means N x the source collection's object count is placed:
# e.g. 1.5 -> the full collection plus its first half copied again.
BUILDING_DENSITY = 1.0
CAR_DENSITY = 1.0
TREE_DENSITY = 0.0
HUMAN_DENSITY = 4.0
ANIMAL_DENSITY = 0.5
OTHER_DENSITY = 1.0
DEBRIS_DENSITY = 1.0

SCATTER_MARGIN_M = 1.0       # meters, extra gap enforced between scattered object footprints
SCATTER_MAX_ATTEMPTS = 200   # rejection-sampling attempts per object before giving up


# list of object collections in the rubbles scene:
"""
Barriers
Humans
Cars
Buildings
Trees
Other
Animals
Debris
Grass
"""
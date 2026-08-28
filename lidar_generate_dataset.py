import blenderproc as bproc
"""
BlenderProc + BLAINDER pipeline: procedurally generate a 50x50m obstacle scene,
fly a solid-state LiDAR (RoboSense E1R) over it with realistic attitude jitter
(non-perfect nadir), and export point cloud + occupancy grid label.

Run with: blenderproc run generate_scene.py <output_dir> <num_scenes>
"""

import bpy
import numpy as np
import random
import os
import sys
import json
import math

try:
    import range_scanner  # BLAINDER addon
except ImportError:
    range_scanner = None  # will fail loudly later if actually invoked

bproc.init()

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "output"
NUM_SCENES = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# Number of LiDAR frames (flight poses) to simulate per scene. The lawn-mower
# flight path is resampled to exactly this many evenly spaced poses.
NUM_FRAMES = 20

AREA_SIZE = 50.0          # meters
GRID_RES = 0.25           # meters per cell -> 200x200 grid
GRID_N = int(AREA_SIZE / GRID_RES)

FLIGHT_ALT_MIN = 30.0
FLIGHT_ALT_MAX = 50.0

# Attitude jitter to simulate imperfect nadir pointing (deg), applied per-pose
MAX_PITCH_DEV_DEG = 8.0     # forward/backward tilt from vertical
MAX_ROLL_DEV_DEG = 8.0      # side-to-side tilt from vertical
MAX_YAW_JITTER_DEG = 5.0    # small heading wander (usually irrelevant for nadir FOV symmetry, kept for realism)

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def create_ground_plane():
    plane = bproc.object.create_primitive("PLANE", scale=[AREA_SIZE / 2, AREA_SIZE / 2, 1])
    plane.set_name("ground")
    mat = bproc.material.create("ground_mat")
    mat.set_principled_shader_value("Base Color", [0.3, 0.28, 0.25, 1.0])
    mat.set_principled_shader_value("Roughness", 0.9)
    plane.replace_materials(mat)
    return plane


def random_obstacle(idx):
    obstacle_type = random.choice(["box", "cylinder", "cone"])
    x = random.uniform(-AREA_SIZE / 2 + 2, AREA_SIZE / 2 - 2)
    y = random.uniform(-AREA_SIZE / 2 + 2, AREA_SIZE / 2 - 2)

    if obstacle_type == "box":
        sx = random.uniform(0.5, 3.0)
        sy = random.uniform(0.5, 3.0)
        sz = random.uniform(0.5, 3.0)
        obj = bproc.object.create_primitive("CUBE", scale=[sx, sy, sz])
    elif obstacle_type == "cylinder":
        r = random.uniform(0.3, 1.5)
        h = random.uniform(0.5, 3.0)
        obj = bproc.object.create_primitive("CYLINDER", scale=[r, r, h])
    else:
        r = random.uniform(0.3, 1.5)
        h = random.uniform(0.5, 3.0)
        obj = bproc.object.create_primitive("CONE", scale=[r, r, h])

    obj.set_location([x, y, obj.get_scale()[2]])  # sit on ground
    obj.set_name(f"obstacle_{idx}")

    mat = bproc.material.create(f"obs_mat_{idx}")
    mat.set_principled_shader_value(
        "Base Color", [random.uniform(0.1, 0.8) for _ in range(3)] + [1.0]
    )
    obj.replace_materials(mat)
    return obj


def obstacle_footprint_radius(obj):
    """Approximate obstacle footprint as bounding circle radius for rasterization."""
    bbox = obj.get_bound_box()
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    cx, cy = obj.get_location()[0], obj.get_location()[1]
    r = max(max(xs) - cx, cx - min(xs), max(ys) - cy, cy - min(ys))
    return cx, cy, r


def rasterize_occupancy(obstacle_info, grid_n=GRID_N, area_size=AREA_SIZE):
    grid = np.zeros((grid_n, grid_n), dtype=np.uint8)
    res = area_size / grid_n
    for (cx, cy, r) in obstacle_info:
        gx = int((cx + area_size / 2) / res)
        gy = int((cy + area_size / 2) / res)
        gr = max(1, int(r / res))
        xmin, xmax = max(0, gx - gr), min(grid_n, gx + gr + 1)
        ymin, ymax = max(0, gy - gr), min(grid_n, gy + gr + 1)
        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                if (i - gx) ** 2 + (j - gy) ** 2 <= gr ** 2:
                    grid[j, i] = 1  # occupied
    return grid


def sample_start_target(occupancy_grid, min_dist=20.0, area_size=AREA_SIZE):
    """Pick start/target in free cells, far apart."""
    grid_n = occupancy_grid.shape[0]
    res = area_size / grid_n
    free_cells = np.argwhere(occupancy_grid == 0)
    sx = sy = tx = ty = 0.0
    for _ in range(200):
        s_idx = free_cells[random.randrange(len(free_cells))]
        t_idx = free_cells[random.randrange(len(free_cells))]
        sx = s_idx[1] * res - area_size / 2
        sy = s_idx[0] * res - area_size / 2
        tx = t_idx[1] * res - area_size / 2
        ty = t_idx[0] * res - area_size / 2
        if np.hypot(tx - sx, ty - sy) >= min_dist:
            return (sx, sy), (tx, ty)
    return (sx, sy), (tx, ty)  # fallback


# ---------------------------------------------------------------------------
# Flight path + attitude jitter
# ---------------------------------------------------------------------------

def build_flight_path(altitude, lane_spacing=8.0, area_size=AREA_SIZE, num_frames=NUM_FRAMES):
    """
    Lawn-mower raster path over area, resampled to exactly num_frames evenly
    spaced poses along the path (so NUM_FRAMES fully controls how many LiDAR
    frames get simulated per scene, independent of lane density).
    """
    half = area_size / 2
    lanes = np.arange(-half + lane_spacing / 2, half, lane_spacing)
    dense_poses = []
    direction = 1
    for y in lanes:
        xs = np.linspace(-half, half, 40)
        if direction < 0:
            xs = xs[::-1]
        for x in xs:
            dense_poses.append([x, y, altitude])
        direction *= -1

    dense_poses = np.array(dense_poses)
    # Resample by cumulative arc length to get num_frames evenly spaced poses
    deltas = np.linalg.norm(np.diff(dense_poses, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(deltas)])
    total_dist = cum_dist[-1] if cum_dist[-1] > 0 else 1.0
    sample_targets = np.linspace(0.0, total_dist, num_frames)
    resampled = np.empty((num_frames, 3))
    for dim in range(3):
        resampled[:, dim] = np.interp(sample_targets, cum_dist, dense_poses[:, dim])
    return resampled.tolist()


def sample_attitude_jitter():
    """
    Returns (pitch_dev_deg, roll_dev_deg, yaw_dev_deg) sampled around a
    nominal nadir attitude, simulating imperfect UAV stabilization
    (wind gusts, control latency, gimbal-less rigid mount, etc.).
    Uses a truncated Gaussian so most poses are close to nadir with
    occasional larger deviations.
    """
    pitch_dev = np.clip(random.gauss(0, MAX_PITCH_DEV_DEG / 3), -MAX_PITCH_DEV_DEG, MAX_PITCH_DEV_DEG)
    roll_dev = np.clip(random.gauss(0, MAX_ROLL_DEV_DEG / 3), -MAX_ROLL_DEV_DEG, MAX_ROLL_DEV_DEG)
    yaw_dev = np.clip(random.gauss(0, MAX_YAW_JITTER_DEG / 3), -MAX_YAW_JITTER_DEG, MAX_YAW_JITTER_DEG)
    return pitch_dev, roll_dev, yaw_dev


def build_flight_attitudes(num_poses, per_pose_jitter=True):
    """
    Generate an attitude (pitch_dev, roll_dev, yaw_dev) for each flight pose.
    If per_pose_jitter=False, one jitter is sampled per scene and held constant
    (simulates a constant trim/mounting offset rather than gust-by-gust wobble).
    """
    if not per_pose_jitter:
        fixed = sample_attitude_jitter()
        return [fixed] * num_poses
    return [sample_attitude_jitter() for _ in range(num_poses)]


# ---------------------------------------------------------------------------
# BLAINDER integration
# ---------------------------------------------------------------------------

def enable_blainder_addon():
    addon_name = "range_scanner"  # module name as installed from the zip
    if addon_name not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module=addon_name)


def create_lidar_scanner_object(name="LidarScanner"):
    """BLAINDER requires the scanner object to be of type CAMERA."""
    cam_data = bpy.data.cameras.new(name=f"{name}_data")
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def set_pose(cam_obj, x, y, z, pitch_dev_deg=0.0, roll_dev_deg=0.0, yaw_dev_deg=0.0):
    """
    Nominal nadir = camera pointing straight down (-Z world).
    A camera object with rotation_euler=(0,0,0) already looks straight down
    along world -Z, so nadir is the identity rotation. Deviations tilt the
    camera off nadir to simulate imperfect UAV attitude (pitch/roll) plus a
    small yaw jitter.
    """
    cam_obj.location = (x, y, z)
    pitch_rad = math.radians(pitch_dev_deg)
    roll_rad = math.radians(roll_dev_deg)
    yaw_rad = math.radians(yaw_dev_deg)
    cam_obj.rotation_euler = (pitch_rad, roll_rad, yaw_rad)


def animate_flight_path(cam_obj, flight_poses, attitudes, frame_step=1):
    """Keyframe the scanner object along the raster flight path with jitter."""
    for i, ((x, y, z), (pdev, rdev, ydev)) in enumerate(zip(flight_poses, attitudes)):
        frame = i * frame_step + 1
        bpy.context.scene.frame_set(frame)
        set_pose(cam_obj, x, y, z, pdev, rdev, ydev)
        cam_obj.keyframe_insert(data_path="location", frame=frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    return len(flight_poses) * frame_step


# --- Sensor preset ---
# RoboSense E1R: fully solid-state digital LiDAR (SPAD-SoC + 2D VCSEL, no
# moving parts), FOV 120(H) x 90(V) deg, angular resolution 0.625(H) x
# 0.625(V) deg -> resX = 120/0.625 = 192, resY = 90/0.625 = 144.
# Source: robosense.ai/en/IncrementalComponents/E1R
ROBOSENSE_E1R_PARAMS = dict(resolutionX=192, fovX=120.0, resolutionY=144, fovY=90.0)


def run_blainder_scan(cam_obj, scene_dir, num_frames, sensor_params):
    if range_scanner is None:
        raise RuntimeError("range_scanner module not importable - check BLAINDER addon installation")

    enable_blainder_addon()

    export_path = os.path.join(scene_dir, "lidar")
    os.makedirs(export_path, exist_ok=True)

    range_scanner.ui.user_interface.scan_static(
        bpy.context,
        scannerObject=cam_obj,
        resolutionX=sensor_params["resolutionX"], fovX=sensor_params["fovX"],
        resolutionY=sensor_params["resolutionY"], fovY=sensor_params["fovY"],
        resolutionPercentage=100,

        reflectivityLower=0.0, distanceLower=0.0,
        reflectivityUpper=0.0, distanceUpper=99999.9, maxReflectionDepth=10,

        enableAnimation=True, frameStart=1, frameEnd=num_frames, frameStep=1, frameRate=1,

        addNoise=True, noiseType='gaussian', mu=0.0, sigma=0.02,
        noiseAbsoluteOffset=0.0, noiseRelativeOffset=0.0,

        simulateRain=False, rainfallRate=0.0,

        addMesh=False,

        exportLAS=False, exportHDF=False, exportCSV=False, exportPLY=True,
        exportSingleFrames=True,   # one .ply per flight pose -> merge later
        dataFilePath=export_path, dataFileName="scan",

        # Newer BLAINDER versions (incl. blender_4.2_lts branch) also require
        # these image-export related args even for point-cloud-only scans.
        exportRenderedImage=False, exportSegmentedImage=False,
        exportPascalVoc=False, exportDepthmap=False,
        depthMinDistance=0.0, depthMaxDistance=100.0,

        debugLines=False, debugOutput=False, outputProgress=True,
        measureTime=False, singleRay=False, destinationObject=None, targetObject=None,
    )

    return export_path


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

for scene_idx in range(NUM_SCENES):
    bproc.clean_up()

    create_ground_plane()

    num_obstacles = random.randint(15, 40)
    obstacle_info = []
    for i in range(num_obstacles):
        obj = random_obstacle(i)
        cx, cy, r = obstacle_footprint_radius(obj)
        obstacle_info.append((cx, cy, r))

    occupancy_grid = rasterize_occupancy(obstacle_info)
    start_xy, target_xy = sample_start_target(occupancy_grid)

    # Light
    light = bproc.types.Light()
    light.set_type("SUN")
    light.set_location([0, 0, 50])
    light.set_energy(5)

    # Flight path
    altitude = random.uniform(FLIGHT_ALT_MIN, FLIGHT_ALT_MAX)
    flight_poses = build_flight_path(altitude)

    # Attitude jitter per pose (set per_pose_jitter=False for a constant
    # scene-wide mounting offset instead of per-pose gust wobble)
    attitudes = build_flight_attitudes(len(flight_poses), per_pose_jitter=True)

    # LiDAR scan via BLAINDER
    lidar_scanner = create_lidar_scanner_object(f"LidarScanner_scene{scene_idx}")
    num_frames = animate_flight_path(lidar_scanner, flight_poses, attitudes)

    lidar_out_dir = run_blainder_scan(lidar_scanner, os.path.join(OUT_DIR, f"scene_{scene_idx:04d}"),
                                       num_frames, ROBOSENSE_E1R_PARAMS)

    # Save occupancy grid + metadata
    scene_dir = os.path.join(OUT_DIR, f"scene_{scene_idx:04d}")
    os.makedirs(scene_dir, exist_ok=True)

    np.save(os.path.join(scene_dir, "occupancy_grid.npy"), occupancy_grid)

    meta = {
        "area_size": AREA_SIZE,
        "grid_res": GRID_RES,
        "grid_n": GRID_N,
        "altitude": altitude,
        "start": start_xy,
        "target": target_xy,
        "num_obstacles": num_obstacles,
        "sensor": "robosense_e1r",
        "sensor_params": ROBOSENSE_E1R_PARAMS,
        "flight_poses": flight_poses,
        "attitude_jitter_deg": attitudes,  # (pitch_dev, roll_dev, yaw_dev) per pose
    }
    with open(os.path.join(scene_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[scene {scene_idx}] sensor=robosense_e1r obstacles={num_obstacles} "
          f"altitude={altitude:.1f}m start={start_xy} target={target_xy}")

print("Done.")

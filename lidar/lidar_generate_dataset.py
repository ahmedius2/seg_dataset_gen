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

# Script usage: blenderproc run lidar_generate_dataset.py <output_dir> <num_scenes>
OUT_DIR = "output"
NUM_SCENES = 10
if len(sys.argv) >= 2:
    OUT_DIR = sys.argv[1]
if len(sys.argv) >= 3:
    NUM_SCENES = int(sys.argv[2])

bproc.init()

# Force Blender to use the NVIDIA RTX 3050 only for rendering to avoid mixed CPU/GPU scheduling.
cycles_preferences = bpy.context.preferences.addons['cycles'].preferences
cycles_preferences.compute_device_type = 'OPTIX'

for device in cycles_preferences.get_devices_for_type('OPTIX'):
    device.use = (device.type == 'OPTIX' and device.name == 'NVIDIA GeForce RTX 3050')

for scene in bpy.data.scenes:
    scene.cycles.device = 'GPU'
    scene.render.engine = 'CYCLES'

print(f"Generating {NUM_SCENES} scenes into {OUT_DIR}")

# Number of LiDAR frames (flight poses) to simulate per scene. The lawn-mower
# flight path is resampled to exactly this many evenly spaced poses.
NUM_FRAMES = 3

PLANE_SIZE = 300.0        # visible ground plane size in meters
AREA_SIZE = 50.0          # meters: inner active region for obstacle generation and flight path
CELL_SIZE = 2.0            # 2x2 m grid cells
CELL_COUNT = int(AREA_SIZE / CELL_SIZE)
GRID_RES = 0.25            # meters per cell -> 200x200 grid
GRID_N = int(AREA_SIZE / GRID_RES)

FLIGHT_ALT_MIN = 30.0
FLIGHT_ALT_MAX = 40.0

# Attitude jitter to simulate imperfect nadir pointing (deg), applied per-pose
MAX_PITCH_DEV_DEG = 8.0     # forward/backward tilt from vertical
MAX_ROLL_DEV_DEG = 8.0      # side-to-side tilt from vertical
MAX_YAW_JITTER_DEG = 5.0    # small heading wander (usually irrelevant for nadir FOV symmetry, kept for realism)

# --- Sensor preset ---
# RoboSense E1R: fully solid-state digital LiDAR (SPAD-SoC + 2D VCSEL, no
# moving parts), FOV 120(H) x 90(V) deg, angular resolution 0.625(H) x
# 0.625(V) deg -> resX = 120/0.625 = 192, resY = 90/0.625 = 144.
# Source: robosense.ai/en/IncrementalComponents/E1R
LIDAR_FOV_X_DEG = 120.0
LIDAR_FOV_Y_DEG = 90.0
ROBOSENSE_E1R_PARAMS = dict(
    resolutionX=192,
    fovX=LIDAR_FOV_X_DEG,
    resolutionY=144,
    fovY=LIDAR_FOV_Y_DEG,
)

# Keep the RGB render camera geometry consistent with the LiDAR's 120x90 deg
# FOV using Blender's explicit camera angle properties.
RENDER_RES_X = 640
RENDER_RES_Y = 480


os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def create_ground_plane():
    plane = bproc.object.create_primitive("PLANE", scale=[PLANE_SIZE / 2, PLANE_SIZE / 2, 1])
    plane.set_name("ground")
    mat = bproc.material.create("ground_mat")
    mat.set_principled_shader_value("Base Color", [0.3, 0.28, 0.25, 1.0])
    mat.set_principled_shader_value("Roughness", 0.9)
    plane.replace_materials(mat)
    return plane


def random_obstacle(idx, x=None, y=None):
    obstacle_type = random.choice(["box", "cylinder", "cone"])
    if x is None:
        x = random.uniform(-AREA_SIZE / 2 + 2, AREA_SIZE / 2 - 2)
    if y is None:
        y = random.uniform(-AREA_SIZE / 2 + 2, AREA_SIZE / 2 - 2)

    if obstacle_type == "box":
        sx = random.uniform(0.6, 1.4)
        sy = random.uniform(0.6, 1.4)
        sz = random.uniform(0.5, 2.2)
        obj = bproc.object.create_primitive("CUBE", scale=[sx, sy, sz])
    elif obstacle_type == "cylinder":
        r = random.uniform(0.35, 0.8)
        h = random.uniform(0.5, 2.0)
        obj = bproc.object.create_primitive("CYLINDER", scale=[r, r, h])
    else:
        r = random.uniform(0.35, 0.8)
        h = random.uniform(0.5, 2.0)
        obj = bproc.object.create_primitive("CONE", scale=[r, r, h])

    obj.set_location([x, y, obj.get_scale()[2] / 2.0])
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


def cell_center_from_index(row, col):
    x = -AREA_SIZE / 2 + CELL_SIZE * (col + 0.5)
    y = -AREA_SIZE / 2 + CELL_SIZE * (row + 0.5)
    return x, y


def cell_distance_meters(a, b):
    ax, ay = cell_center_from_index(*a)
    bx, by = cell_center_from_index(*b)
    return math.hypot(bx - ax, by - ay)


# ----
# Smooth spline-based flight corridors (replaces BFS shortest paths)
# ----

def _catmull_rom_spline(pts, samples=400):
    """Smooth C1-continuous curve passing through all control points (x=col, y=row)."""
    pts = np.asarray(pts, dtype=float)
    p = np.vstack([pts[0], pts, pts[-1]])          # clamp endpoints
    n_seg = len(p) - 3
    per = max(2, samples // max(1, n_seg))
    out = []
    for i in range(n_seg):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        for j in range(per):
            t = j / per
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1)
                              + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-2])
    return np.array(out)


def _smooth_corridor(start, target, lateral_bias=0.0, wobble=0.35,
                     num_ctrl=6, rng=random):
    """Generate a gently curving corridor from start to target.

    lateral_bias in [-1, 1] biases the whole route to bow left/right of the
    straight line; wobble adds mild per-control-point randomness. The bow
    envelope is zero at the endpoints (sin(pi*t)) so start/target stay fixed.
    """
    (r0, c0), (r1, c1) = start, target
    a = np.array([c0, r0], float)
    b = np.array([c1, r1], float)
    seg = b - a
    length = np.linalg.norm(seg)
    if length < 1e-6:
        return [start]
    perp = np.array([-seg[1], seg[0]]) / length     # unit perpendicular

    ctrl = []
    for i in range(num_ctrl):
        t = i / (num_ctrl - 1)
        env = math.sin(math.pi * t)                 # 0 at ends, 1 in middle
        offset = (lateral_bias * env * CELL_COUNT * 0.20
                  + rng.uniform(-wobble, wobble) * env * CELL_COUNT * 0.10)
        ctrl.append(a + seg * t + perp * offset)
    return _catmull_rom_spline(np.array(ctrl))


def _curve_to_cells(curve):
    """Rasterize a continuous curve to a 4-connected list of grid cells."""
    ordered, seen, last = [], set(), None
    for x, y in curve:
        r, c = int(round(y)), int(round(x))
        if not (0 <= r < CELL_COUNT and 0 <= c < CELL_COUNT):
            continue
        cell = (r, c)
        if cell == last:
            continue
        # bridge diagonal jumps with an L step to keep 4-connectivity
        if last is not None and abs(cell[0] - last[0]) + abs(cell[1] - last[1]) > 1:
            bridge = (last[0], cell[1])
            if bridge not in seen:
                seen.add(bridge); ordered.append(bridge)
        if cell not in seen:
            seen.add(cell); ordered.append(cell)
        last = cell
    return ordered


def generate_cell_paths(start, target, min_paths=2, max_paths=5, rng=random):
    """Generate several smooth, visually distinct corridors from start to target.

    Distinct routes are produced by giving each a different lateral bias so they
    bow to different sides; interior cells are kept non-overlapping.
    """
    n = rng.randint(min_paths, max_paths)
    biases = list(np.linspace(-1.0, 1.0, n))
    rng.shuffle(biases)

    paths, used = [], set()
    for bias in biases:
        for _ in range(8):                          # a few tries to avoid overlap
            cells = _curve_to_cells(_smooth_corridor(start, target, bias, rng=rng))
            if len(cells) < 2:
                bias += rng.uniform(-0.2, 0.2)
                continue
            if cells[0] != start:
                cells = [start] + cells
            if cells[-1] != target:
                cells = cells + [target]
            interior = set(cells[1:-1])
            if interior & used:                     # overlaps another route -> nudge & retry
                bias = bias * 0.75 + rng.uniform(-0.2, 0.2)
                continue
            paths.append(cells)
            used.update(interior)
            break

    return paths if len(paths) >= min_paths else []

def sample_start_target_in_grid(min_dist=40.0):
    """Pick a start and goal cell that are far apart and likely connected."""
    for _ in range(5000):
        start = (random.randrange(CELL_COUNT), random.randrange(CELL_COUNT))
        target = (random.randrange(CELL_COUNT), random.randrange(CELL_COUNT))
        if cell_distance_meters(start, target) >= min_dist:
            return start, target
    raise RuntimeError("Unable to sample valid start/target cells for the 2x2 m grid.")



# ----
# Organic rubble field: open floor + clustered debris + wall fragments
# ----

def _corridor_clearance(paths, radius_cells=1):
    """All corridor cells plus a buffer, so lanes stay comfortably wide/open."""
    free = set()
    for path in paths:
        for (r, c) in path:
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < CELL_COUNT and 0 <= cc < CELL_COUNT:
                        free.add((rr, cc))
    return free


def _grow_blob(seed, size, forbidden, occupied, rng=random):
    """Grow an irregular connected cluster of cells from a seed (rubble pile)."""
    blob = {seed}
    frontier = [seed]
    while len(blob) < size and frontier:
        r, c = frontier[rng.randrange(len(frontier))]
        nbrs = [(r+dr, c+dc) for dr, dc in
                [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]]
        rng.shuffle(nbrs)
        added = False
        for rr, cc in nbrs:
            if (0 <= rr < CELL_COUNT and 0 <= cc < CELL_COUNT
                    and (rr, cc) not in forbidden and (rr, cc) not in occupied
                    and (rr, cc) not in blob):
                blob.add((rr, cc)); frontier.append((rr, cc)); added = True
                break
        if not added:
            frontier.remove((r, c))
    return blob


def _make_wall(forbidden, occupied, rng=random):
    """A long, thin, slightly bent wall fragment (like the toppled slabs)."""
    length = rng.randint(4, 8)
    r = rng.randrange(CELL_COUNT); c = rng.randrange(CELL_COUNT)
    horiz = rng.random() < 0.5
    cells = []
    for i in range(length):
        rr = r + (0 if horiz else i) + (rng.randint(-1, 1) if i and rng.random() < 0.25 else 0)
        cc = c + (i if horiz else 0) + (rng.randint(-1, 1) if i and rng.random() < 0.25 else 0)
        if (0 <= rr < CELL_COUNT and 0 <= cc < CELL_COUNT
                and (rr, cc) not in forbidden and (rr, cc) not in occupied):
            cells.append((rr, cc))
    return cells

# ----
# Poisson-disk (blue-noise) cluster seeding with halo separation
# ----

def _poisson_seeds(k, min_dist, forbidden, occupied, rng=random):
    """Blue-noise seeds >= min_dist apart, avoiding corridors & existing debris.

    Simple Bridson-style rejection on the discrete grid: shuffle candidate
    cells and greedily accept ones that respect the minimum spacing.
    """
    seeds = []
    cand = [(r, c) for r in range(CELL_COUNT) for c in range(CELL_COUNT)
            if (r, c) not in forbidden and (r, c) not in occupied]
    rng.shuffle(cand)
    md2 = min_dist * min_dist
    for (r, c) in cand:
        if len(seeds) >= k:
            break
        if all((r - sr) ** 2 + (c - sc) ** 2 >= md2 for sr, sc in seeds):
            seeds.append((r, c))
    return seeds


def _halo(cells, radius=1):
    """1-cell (or more) ring of cells surrounding a set, for separation."""
    ring = set()
    for (r, c) in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < CELL_COUNT and 0 <= cc < CELL_COUNT:
                    ring.add((rr, cc))
    return ring - set(cells)


def build_cell_occupancy(paths, num_clusters=10, num_walls=5,
                         cluster_min=15, cluster_max=25,
                         min_cluster_dist=5.0, halo_radius=1, rng=random):
    """Open arena floor with well-spread rubble clusters + wall fragments.

    - Corridors (plus clearance buffer) are guaranteed clear.
    - Cluster seeds are placed via blue-noise so they spread evenly.
    - A halo ring is added to `forbidden` after each cluster so clusters
      cannot touch or fuse; sizes are honored because blobs grow into space
      that is known to be clear.
    """
    forbidden = _corridor_clearance(paths, radius_cells=1)
    occupied = set()

    # Walls first (long thin slabs), then fence them off with a halo too.
    for _ in range(num_walls):
        wall = _make_wall(forbidden, occupied, rng)
        if wall:
            occupied.update(wall)
            forbidden |= _halo(wall, halo_radius)

    # Blue-noise seeds for cluster centers, respecting spacing.
    seeds = _poisson_seeds(num_clusters, min_cluster_dist, forbidden, occupied, rng)

    for seed in seeds:
        if seed in forbidden or seed in occupied:
            continue
        size = rng.randint(cluster_min, cluster_max)
        blob = _grow_blob(seed, size, forbidden, occupied, rng)
        occupied.update(blob)
        # Reserve a ring around this cluster so the next one can't fuse into it.
        forbidden |= _halo(blob, halo_radius)

    all_cells = {(r, c) for r in range(CELL_COUNT) for c in range(CELL_COUNT)}
    free_cells = all_cells - occupied
    return free_cells, occupied

def rasterize_cell_occupancy(occupied_cells, cell_size=CELL_SIZE, area_size=AREA_SIZE, grid_n=GRID_N):
    grid = np.zeros((grid_n, grid_n), dtype=np.uint8)
    res = area_size / grid_n
    cell_steps = int(round(cell_size / res))

    for row, col in occupied_cells:
        x0 = -area_size / 2 + col * cell_size
        y0 = -area_size / 2 + row * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size

        x_min = max(0, int(np.floor((x0 + area_size / 2) / res)))
        x_max = min(grid_n, int(np.ceil((x1 + area_size / 2) / res)))
        y_min = max(0, int(np.floor((y0 + area_size / 2) / res)))
        y_max = min(grid_n, int(np.ceil((y1 + area_size / 2) / res)))

        grid[y_min:y_max, x_min:x_max] = 1

    return grid


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
    cam_data.lens = 35.0
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def create_render_camera_object(name="RenderCamera"):
    """Standard Blender camera for RGB scene render; match the LiDAR FOV exactly."""
    cam_data = bpy.data.cameras.new(name=f"{name}_data")
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.angle_x = math.radians(LIDAR_FOV_X_DEG)
    cam_data.angle_y = math.radians(LIDAR_FOV_Y_DEG)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def set_active_render_camera(cam_obj):
    bpy.context.scene.camera = cam_obj
    bpy.context.view_layer.objects.active = cam_obj
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


def render_camera_frames(cam_obj, scene_dir, flight_poses, attitudes, image_dir_name="camera"):
    """Save RGB camera frames immediately after the LiDAR scan, aligned with the scan frame index."""
    image_dir = os.path.join(scene_dir, image_dir_name)
    os.makedirs(image_dir, exist_ok=True)

    set_active_render_camera(cam_obj)

    scene = bpy.context.scene
    old_file_format = scene.render.image_settings.file_format
    old_color_mode = scene.render.image_settings.color_mode
    old_resolution_x = scene.render.resolution_x
    old_resolution_y = scene.render.resolution_y
    old_percentage = scene.render.resolution_percentage
    old_engine = scene.render.engine
    old_background = scene.world.color if scene.world else None

    # Cycles samples = Monte Carlo rays per pixel for the image. More samples reduce
    # noise but increase render time roughly linearly. For fast dataset generation,
    # we keep it much lower than the default 1024.
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x = RENDER_RES_X
    scene.render.resolution_y = RENDER_RES_Y
    scene.render.resolution_percentage = 100
    scene.render.engine = 'CYCLES'
    scene.render.film_transparent = False
    scene.cycles.samples = 64
    scene.cycles.use_adaptive_sampling = False

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.8, 0.8, 0.8)

    try:
        for frame_idx, ((x, y, z), (pdev, rdev, ydev)) in enumerate(zip(flight_poses, attitudes), start=1):
            scene.frame_set(frame_idx)
            set_pose(cam_obj, x, y, z, pdev, rdev, ydev)
            scene.render.filepath = os.path.join(image_dir, f"frame_{frame_idx:04d}.png")
            bpy.ops.render.render(write_still=True)
    finally:
        scene.render.image_settings.file_format = old_file_format
        scene.render.image_settings.color_mode = old_color_mode
        scene.render.resolution_x = old_resolution_x
        scene.render.resolution_y = old_resolution_y
        scene.render.resolution_percentage = old_percentage
        scene.render.engine = old_engine
        scene.render.film_transparent = False
        if old_background is not None and scene.world is not None:
            scene.world.color = old_background

    return image_dir


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

for scene_idx in range(NUM_SCENES):
    bproc.clean_up()

    create_ground_plane()

    sun = bproc.types.Light()
    sun.set_type("SUN")
    sun.set_location([0, 0, 50])
    sun.set_energy(5)
    sun.set_rotation_euler([math.radians(60), 0.0, math.radians(45)])

    fill = bproc.types.Light()
    fill.set_type("AREA")
    fill.set_location([10, -10, 12])
    fill.set_energy(3000)
    fill.set_scale([4, 4, 4])

    rim = bproc.types.Light()
    rim.set_type("AREA")
    rim.set_location([-12, 8, 12])
    rim.set_energy(2500)
    rim.set_scale([4, 4, 4])

    for _ in range(200):
        start_cell, target_cell = sample_start_target_in_grid(min_dist=40.0)
        paths = generate_cell_paths(start_cell, target_cell, min_paths=2, max_paths=5)
        if len(paths) >= 2:
            break
    else:
        raise RuntimeError(f"Unable to find a valid start/target pair with at least two non-overlapping paths for scene {scene_idx}.")

    free_cells, occupied_cells = build_cell_occupancy(paths)

    obstacle_info = []
    cell_obstacles = sorted(occupied_cells)
    for idx, (row, col) in enumerate(cell_obstacles):
        x, y = cell_center_from_index(row, col)
        obj = random_obstacle(idx, x=x, y=y)
        cx, cy, r = obstacle_footprint_radius(obj)
        obstacle_info.append((cx, cy, r))

    occupancy_grid = rasterize_cell_occupancy(occupied_cells)
    start_xy = cell_center_from_index(*start_cell)
    target_xy = cell_center_from_index(*target_cell)
    num_obstacles = len(cell_obstacles)

    # Flight path
    altitude = random.uniform(FLIGHT_ALT_MIN, FLIGHT_ALT_MAX)
    flight_poses = build_flight_path(altitude)

    # Attitude jitter per pose (set per_pose_jitter=False for a constant
    # scene-wide mounting offset instead of per-pose gust wobble)
    attitudes = build_flight_attitudes(len(flight_poses), per_pose_jitter=True)

    # LiDAR scan via BLAINDER
    lidar_scanner = create_lidar_scanner_object(f"LidarScanner_scene{scene_idx}")
    render_camera = create_render_camera_object(f"RenderCamera_scene{scene_idx}")
    num_frames = animate_flight_path(lidar_scanner, flight_poses, attitudes)

    scene_dir = os.path.join(OUT_DIR, f"scene_{scene_idx:04d}")
    os.makedirs(scene_dir, exist_ok=True)

    render_camera_frames(render_camera, scene_dir, flight_poses, attitudes, image_dir_name="camera")
    run_blainder_scan(lidar_scanner, scene_dir, num_frames, ROBOSENSE_E1R_PARAMS)

    # Save occupancy grid + metadata

    np.save(os.path.join(scene_dir, "occupancy_grid.npy"), occupancy_grid)

    meta = {
        "plane_size": PLANE_SIZE,
        "area_size": AREA_SIZE,
        "cell_size": CELL_SIZE,
        "cell_count": CELL_COUNT,
        "grid_res": GRID_RES,
        "grid_n": GRID_N,
        "altitude": altitude,
        "start_cell": list(start_cell),
        "target_cell": list(target_cell),
        "start": start_xy,
        "target": target_xy,
        "paths": [path for path in paths],
        "num_paths": len(paths),
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

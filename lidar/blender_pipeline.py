import numpy as np
import math
import os
import random
import blenderproc as bproc
import bpy
import bmesh
from mathutils import noise as bnoise

from dataset_config import (
    AREA_SIZE_M,
    CELL_SIZE_M,
    GROUND_NOISE_AMPLITUDE_M,
    GROUND_NOISE_CELL_SIZE_M,
    GROUND_NOISE_MICRO_STD_M,
    GROUND_NOISE_SCALE,
    LIDAR_FOV_X_DEG,
    LIDAR_FOV_Y_DEG,
    MAX_PITCH_DEV_DEG,
    MAX_ROLL_DEV_DEG,
    MAX_YAW_JITTER_DEG,
    NUM_FRAMES,
)

try:
    import range_scanner
except ImportError:
    range_scanner = None


def create_noisy_ground_mesh(
    name,
    size_m,
    cell_size_m=GROUND_NOISE_CELL_SIZE_M,
    amplitude_m=GROUND_NOISE_AMPLITUDE_M,
    noise_scale=GROUND_NOISE_SCALE,
    micro_std_m=GROUND_NOISE_MICRO_STD_M,
    rng=random,
):
    """Build a subdivided ground plane with Perlin undulation + per-vertex jitter.

    Replaces a perfectly flat plane with a rough terrain surface so it looks
    (and scans) more realistically.
    """


    segments = max(2, int(round(size_m / cell_size_m)))
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=segments, y_segments=segments, size=size_m / 2.0)

    # Random offset so every scene gets a differently seeded noise pattern.
    ox, oy = rng.uniform(0.0, 1000.0), rng.uniform(0.0, 1000.0)
    for v in bm.verts:
        n = bnoise.noise((v.co.x * noise_scale + ox, v.co.y * noise_scale + oy, 0.0)) - 0.5
        v.co.z += n * 2.0 * amplitude_m
        v.co.z += rng.gauss(0.0, micro_std_m)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    blender_obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(blender_obj)
    bpy.context.view_layer.objects.active = blender_obj
    blender_obj.select_set(True)
    bpy.ops.rigidbody.object_add()
    blender_obj.rigid_body.type = 'PASSIVE'
    blender_obj.select_set(False)
    return bproc.types.MeshObject(blender_obj)

def cell_center_from_index(row, col):
    x = -AREA_SIZE_M / 2 + CELL_SIZE_M * (col + 0.5)
    y = -AREA_SIZE_M / 2 + CELL_SIZE_M * (row + 0.5)
    return x, y


def build_flight_path(altitude, lane_spacing=8.0, area_size=AREA_SIZE_M, num_frames=NUM_FRAMES):
    """Lawn-mower raster path resampled to exactly num_frames evenly spaced poses."""
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
    deltas = np.linalg.norm(np.diff(dense_poses, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(deltas)])
    total_dist = cum_dist[-1] if cum_dist[-1] > 0 else 1.0
    sample_targets = np.linspace(0.0, total_dist, num_frames)
    resampled = np.empty((num_frames, 3))
    for dim in range(3):
        resampled[:, dim] = np.interp(sample_targets, cum_dist, dense_poses[:, dim])
    return resampled.tolist()


def sample_attitude_jitter():
    """Sample small pose-to-pose attitude jitter around nominal nadir."""
    pitch_dev = np.clip(random.gauss(0, MAX_PITCH_DEV_DEG / 3), -MAX_PITCH_DEV_DEG, MAX_PITCH_DEV_DEG)
    roll_dev = np.clip(random.gauss(0, MAX_ROLL_DEV_DEG / 3), -MAX_ROLL_DEV_DEG, MAX_ROLL_DEV_DEG)
    yaw_dev = np.clip(random.gauss(0, MAX_YAW_JITTER_DEG / 3), -MAX_YAW_JITTER_DEG, MAX_YAW_JITTER_DEG)
    return pitch_dev, roll_dev, yaw_dev


def build_flight_attitudes(num_poses, per_pose_jitter=True):
    if not per_pose_jitter:
        fixed = sample_attitude_jitter()
        return [fixed] * num_poses
    return [sample_attitude_jitter() for _ in range(num_poses)]


def enable_blainder_addon():
    addon_name = "range_scanner"
    if addon_name not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module=addon_name)


def create_lidar_scanner_object(name="LidarScanner"):
    cam_data = bpy.data.cameras.new(name=f"{name}_data")
    cam_data.lens = 35.0
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def create_render_camera_object(name="RenderCamera"):
    cam_data = bpy.data.cameras.new(name=f"{name}_data")
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.angle_x = math.radians(LIDAR_FOV_X_DEG)
    cam_data.angle_y = math.radians(LIDAR_FOV_Y_DEG)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def set_pose(cam_obj, x, y, z, pitch_dev_deg=0.0, roll_dev_deg=0.0, yaw_dev_deg=0.0):
    cam_obj.location = (x, y, z)
    pitch_rad = math.radians(pitch_dev_deg)
    roll_rad = math.radians(roll_dev_deg)
    yaw_rad = math.radians(yaw_dev_deg)
    cam_obj.rotation_euler = (pitch_rad, roll_rad, yaw_rad)


def animate_flight_path(cam_obj, flight_poses, attitudes, frame_step=1):
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
    export_path = os.path.join(scene_dir, "lidar_and_camera")
    os.makedirs(export_path, exist_ok=True)

    # exportRenderedImage uses bpy.ops.render.render, which needs an active scene camera
    bpy.context.scene.camera = cam_obj

    range_scanner.ui.user_interface.scan_static(
        bpy.context,
        scannerObject=cam_obj,
        resolutionX=sensor_params["resolutionX"], fovX=sensor_params["fovX"],
        resolutionY=sensor_params["resolutionY"], fovY=sensor_params["fovY"],
        resolutionPercentage=100,
        reflectivityLower=0.0, distanceLower=0.0,
        reflectivityUpper=0.0, distanceUpper=9999.9, maxReflectionDepth=10,
        enableAnimation=True, frameStart=1, frameEnd=num_frames, frameStep=1, frameRate=1,
        addNoise=True, noiseType='gaussian', mu=0.0, sigma=0.02,
        noiseAbsoluteOffset=0.0, noiseRelativeOffset=0.0,
        simulateRain=False, rainfallRate=0.0,
        addMesh=False,
        exportLAS=False, exportHDF=False, exportCSV=False, exportPLY=True,
        exportSingleFrames=True,
        dataFilePath=export_path, dataFileName="scan",
        exportRenderedImage=True, exportSegmentedImage=True,
        exportPascalVoc=False, exportDepthmap=False,
        depthMinDistance=0.0, depthMaxDistance=100.0,
        debugLines=False, debugOutput=False, outputProgress=True,
        measureTime=False, singleRay=False, destinationObject=None, targetObject=None,
    )
    return export_path


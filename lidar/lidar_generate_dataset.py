import blenderproc as bproc

"""BlenderProc dataset generation entrypoint.

This file is intentionally small: the scene-generation logic and LiDAR pipeline
live in sibling modules so the project remains easier to maintain.
"""

import bpy
import numpy as np
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from blender_pipeline import (
    animate_flight_path,
    build_flight_attitudes,
    build_flight_path,
    create_lidar_scanner_object,
    create_render_camera_object,
    create_noisy_ground_mesh,
    run_blainder_scan,
)
from dataset_config import (
    AREA_SIZE_M,
    CELL_SIZE_M,
    FLIGHT_ALT_MAX,
    FLIGHT_ALT_MIN,
    GRID_N,
    LIDAR_SENSOR_NAME,
    LIDAR_SENSOR_PARAMS,
    MASK_DIR,
    MASK_PX,
    NUM_SCENES,
    NUM_SCENES_TO_EXPORT_BLEND,
    OUT_DIR,
    PLANE_SIZE,
    SOURCE_SCENE_PATH
)
from mask_scene import (
    build_scene_from_mask,
    list_mask_files,
    save_occupancy_grid_preview,
)


def configure_blender_gpu():
    """Use the Blender Cycles GPU if available, mirroring the original script."""
    cycles_preferences = bpy.context.preferences.addons['cycles'].preferences
    cycles_preferences.compute_device_type = 'OPTIX'
    for device in cycles_preferences.get_devices_for_type('OPTIX'):
        device.use = (device.type == 'OPTIX' and device.name == 'NVIDIA GeForce RTX 3050')
    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'
        scene.render.engine = 'CYCLES'


def main():
    bproc.init()
    configure_blender_gpu()

    os.makedirs(OUT_DIR, exist_ok=True)
    mask_files = list_mask_files(MASK_DIR)
    if not mask_files:
        raise RuntimeError(
            f"No mask images found in '{MASK_DIR}'. Add {MASK_PX}x{MASK_PX} PNG masks "
            f"(black=rubble, red=barricade, white=clear) to that directory."
        )
    mask_files = mask_files[:NUM_SCENES]
    print(f"Found {len(mask_files)} mask(s); generating one scene per mask.")

    for scene_idx, mask_path in enumerate(mask_files):
        bproc.clean_up()
        print(f"[scene {scene_idx}] reconstructing from mask: {mask_path}")

        scene = build_scene_from_mask(mask_path, SOURCE_SCENE_PATH, rng=random)

        # randomize sun location and energy to simulate different times of day and lighting conditions
        sun = bproc.types.Light()
        sun.set_type("SUN")
        sun.set_location([0, 0, 10])
        sun.set_energy(5)
        sun.set_rotation_euler([math.radians(60), 0.0, math.radians(45)])

        start_px = scene["start_px"]
        target_px = scene["target_px"]
        start_xy = list(scene["start_world"])
        target_xy = list(scene["target_world"])
        occupancy_grid = scene["occupancy_grid"]
        rubble_info = scene["rubble_info"]
        barricade_info = scene["barricade_info"]
        num_obstacles = len(rubble_info) + len(barricade_info)

        print(f"[scene {scene_idx}] start_px={start_px} target_px={target_px} "
              f"rubble={len(rubble_info)} barricades={len(barricade_info)} "
              f"cleared_barricade={scene['cleared_barricade_index']}")

        base_ground = create_noisy_ground_mesh("ground_base", PLANE_SIZE, rng=random)
        base_ground.set_location([0.0, 0.0, -0.05])
        base_mat = bproc.material.create("ground_base_mat")
        base_mat.set_principled_shader_value("Base Color", [0.3, 0.28, 0.25, 1.0])
        base_mat.set_principled_shader_value("Roughness", 0.9)
        base_ground.replace_materials(base_mat)

        for label, (wx, wy), color in [
            ("start_cell", start_xy, [0.15, 0.8, 0.25, 1.0]),
            ("target_cell", target_xy, [0.9, 0.18, 0.18, 1.0]),
        ]:
            marker = bproc.object.create_primitive("PLANE")
            marker.set_name(label)
            marker.set_location([wx, wy, 0.01])
            mmat = bproc.material.create(f"{label}_mat_{scene_idx}")
            mmat.set_principled_shader_value("Base Color", color)
            mmat.set_principled_shader_value("Roughness", 0.9)
            marker.replace_materials(mmat)

        # settled_count = settle_generated_obstacles()
        # print(f"[scene {scene_idx}] settled and froze {settled_count} generated obstacle(s)")

        scene_dir = os.path.join(OUT_DIR, f"scene_{scene_idx:04d}")
        os.makedirs(scene_dir, exist_ok=True)

        if scene_idx < NUM_SCENES_TO_EXPORT_BLEND:
            blend_path = os.path.join(scene_dir, f"scene_{scene_idx:04d}.blend")
            bpy.ops.wm.save_as_mainfile(filepath=blend_path, copy=True)
            print(f"[scene {scene_idx}] exported blend file: {blend_path}")

        print(f"Scene reconstructed from mask. Occupancy grid shape: {occupancy_grid.shape}")

        altitude = random.uniform(FLIGHT_ALT_MIN, FLIGHT_ALT_MAX)
        flight_poses = build_flight_path(altitude)
        attitudes = build_flight_attitudes(len(flight_poses), per_pose_jitter=True)

        lidar_scanner = create_lidar_scanner_object(f"LidarScanner_scene{scene_idx}")
        # render_camera = create_render_camera_object(f"RenderCamera_scene{scene_idx}")
        num_frames = animate_flight_path(lidar_scanner, flight_poses, attitudes)

        # render_camera_frames(render_camera, scene_dir, flight_poses, attitudes, image_dir_name="camera")
        run_blainder_scan(lidar_scanner, scene_dir, num_frames, LIDAR_SENSOR_PARAMS)

        np.save(os.path.join(scene_dir, "occupancy_grid.npy"), occupancy_grid)
        start_gpx = int((start_px[1] / MASK_PX) * GRID_N)
        start_gpy = int(((MASK_PX - 1 - start_px[0]) / MASK_PX) * GRID_N)
        target_gpx = int((target_px[1] / MASK_PX) * GRID_N)
        target_gpy = int(((MASK_PX - 1 - target_px[0]) / MASK_PX) * GRID_N)
        save_occupancy_grid_preview(
            occupancy_grid,
            os.path.join(scene_dir, "occupancy_grid.png"),
            (start_gpy, start_gpx),
            (target_gpy, target_gpx),
        )

        meta = {
            "plane_size": PLANE_SIZE,
            "area_size": AREA_SIZE_M,
            "cell_size": CELL_SIZE_M,
            "cell_count": int(AREA_SIZE_M / CELL_SIZE_M),
            "grid_n": GRID_N,
            "altitude": altitude,
            "mask_path": mask_path,
            "mask_px": MASK_PX,
            "pix_size": AREA_SIZE_M / MASK_PX,
            "start_px": [int(start_px[0]), int(start_px[1])],
            "target_px": [int(target_px[0]), int(target_px[1])],
            "start": start_xy,
            "target": target_xy,
            "paths": [],
            "num_paths": 0,
            "num_obstacles": num_obstacles,
            "num_rubble_piles": len(rubble_info),
            "num_barricades": len(barricade_info),
            "cleared_barricade_index": scene["cleared_barricade_index"],
            "rubble_info": rubble_info,
            "barricade_info": barricade_info,
            "sensor": LIDAR_SENSOR_NAME,
            "sensor_params": LIDAR_SENSOR_PARAMS,
            "flight_poses": flight_poses,
            "attitude_jitter_deg": attitudes,
        }
        with open(os.path.join(scene_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[scene {scene_idx}] sensor={LIDAR_SENSOR_NAME} obstacles={num_obstacles} "
              f"altitude={altitude:.1f}m start={start_xy} target={target_xy}")



    print("Done.")


if __name__ == "__main__":
    main()

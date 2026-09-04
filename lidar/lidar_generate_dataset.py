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
    create_noisy_ground_mesh,
    run_blainder_scan,
)
from dataset_config import (
    AREA_SIZE_M,
    CELL_SIZE_M,
    FLIGHT_ALT_MAX,
    FLIGHT_ALT_MIN,
    GRID_N,
    GROUND_NOISE_CELL_SIZE_CHOICES,
    GROUND_NOISE_VALUE_MAX,
    GROUND_NOISE_VALUE_MIN,
    LIDAR_SENSOR_NAME,
    LIDAR_SENSOR_PARAMS,
    MASK_DIR,
    MASK_MERGE_DIR,
    MASK_MERGE_MODE,
    MASK_PX,
    NUM_SCENES_PER_MASK,
    NUM_SCENES_TO_EXPORT_BLEND,
    OUT_DIR,
    PLANE_SIZE,
    SEED,
    SKIP_RENDER_AND_SCAN,
    SOURCE_SCENE_PATH
)
from mask_scene import (
    build_merged_masks,
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


def parse_seed_arg():
    # allow overriding the config default via `blenderproc run ... -- --seed=123`
    for arg in sys.argv:
        if arg.startswith("--seed="):
            return int(arg.split("=", 1)[1])
    return SEED


def main():
    # allow overriding the config default via `blenderproc run ... -- --skip-render`
    skip_render_and_scan = SKIP_RENDER_AND_SCAN or "--skip-render" in sys.argv

    seed = parse_seed_arg()
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using seed={seed} for deterministic scene generation.")

    bproc.init()
    configure_blender_gpu()

    scenes_dir = os.path.join(OUT_DIR, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)
    mask_files = list_mask_files(MASK_DIR)
    if not mask_files:
        raise RuntimeError(
            f"No mask images found in '{MASK_DIR}'. Add {MASK_PX}x{MASK_PX} PNG masks "
            f"(black=rubble, red=barricade, white=clear) to that directory."
        )
    if MASK_MERGE_MODE:
        # Keep the original masks in play alongside the merged ones.
        mask_files = mask_files + build_merged_masks(mask_files, MASK_MERGE_DIR, rng=random)
    print(f"Found {len(mask_files)} mask(s); generating {NUM_SCENES_PER_MASK} scene(s) per mask.")

    # amplitude/scale/micro-std ramp together, linearly, over the whole run
    ground_noise_values = np.linspace(GROUND_NOISE_VALUE_MIN, GROUND_NOISE_VALUE_MAX, 10)

    mask_idx = 0
    for mask_path in mask_files:
      for scene_idx in range(NUM_SCENES_PER_MASK):
        bproc.clean_up()
        print(f"[scene {mask_idx} - {scene_idx}] reconstructing from mask: {mask_path}")

        global_scene_idx = mask_idx * NUM_SCENES_PER_MASK + scene_idx
        ground_noise_value = float(ground_noise_values[global_scene_idx % len(ground_noise_values)])
        ground_noise_cell_size = GROUND_NOISE_CELL_SIZE_CHOICES[global_scene_idx % len(GROUND_NOISE_CELL_SIZE_CHOICES)]

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
        scatter_info = scene["scatter_info"]

        print(f"[scene {mask_idx} - {scene_idx}] start_px={start_px} target_px={target_px} "
              f"rubble={len(rubble_info)} barricades={len(barricade_info)} "
              f"cleared_barricade={scene['cleared_barricade_index']}")
        print(f"[scene {mask_idx} - {scene_idx}] scattered: "
              + ", ".join(f"{name}={len(items)}" for name, items in scatter_info.items()))

        base_ground = create_noisy_ground_mesh(
            "ground_base", PLANE_SIZE,
            cell_size_m=ground_noise_cell_size,
            amplitude_m=ground_noise_value,
            noise_scale=ground_noise_value,
            micro_std_m=ground_noise_value,
            rng=random,
        )
        base_ground.set_location([0.0, 0.0, -0.05])

        # Use the material named coast_sand_01 already available in the source scene.
        coast_sand_mat_bpy = bpy.data.materials.get("coast_sand_01")
        if coast_sand_mat_bpy:
            base_ground.replace_materials(bproc.types.Material(coast_sand_mat_bpy))
        else:
            base_mat = bproc.material.create("ground_base_mat")
            base_mat.set_principled_shader_value("Base Color", [0.3, 0.28, 0.25, 1.0])
            base_mat.set_principled_shader_value("Roughness", 0.9)
            base_ground.replace_materials(base_mat)

        # dont' place a start/target marker in the scene for now.
        # for label, (wx, wy), color in [
        #     ("start_cell", start_xy, [0.15, 0.8, 0.25, 1.0]),
        #     ("target_cell", target_xy, [0.9, 0.18, 0.18, 1.0]),
        # ]:
        #     marker = bproc.object.create_primitive("PLANE")
        #     marker.set_name(label)
        #     marker.set_location([wx, wy, 0.01])
        #     mmat = bproc.material.create(f"{label}_mat_{mask_idx}")
        #     mmat.set_principled_shader_value("Base Color", color)
        #     mmat.set_principled_shader_value("Roughness", 0.9)
        #     marker.replace_materials(mmat)

        if skip_render_and_scan:
            print(f"[scene {mask_idx} - {scene_idx}] skipping render/scan (SKIP_RENDER_AND_SCAN enabled)")
        else:
            scene_dir = os.path.join(OUT_DIR, f"scene_{mask_idx:04d}")
            os.makedirs(scene_dir, exist_ok=True)

            # settled_count = settle_generated_obstacles()
            # print(f"[scene {scene_idx}] settled and froze {settled_count} generated obstacle(s)")


        if NUM_SCENES_TO_EXPORT_BLEND == -1 or (NUM_SCENES_TO_EXPORT_BLEND > 0 and mask_idx < NUM_SCENES_TO_EXPORT_BLEND):
            blend_path = os.path.join(scenes_dir, f"scene_{mask_idx:04d}_{scene_idx}.blend")
            bpy.ops.wm.save_as_mainfile(filepath=blend_path, copy=True)
            print(f"[scene {mask_idx} - {scene_idx}] exported blend file: {blend_path}")

        print(f"Scene reconstructed from mask. Occupancy grid shape: {occupancy_grid.shape}")

        if not skip_render_and_scan:
            altitude = random.uniform(FLIGHT_ALT_MIN, FLIGHT_ALT_MAX)
            flight_poses = build_flight_path(altitude)
            attitudes = build_flight_attitudes(len(flight_poses), per_pose_jitter=True)

            lidar_scanner = create_lidar_scanner_object(f"LidarScanner_scene{mask_idx}")
            num_frames = animate_flight_path(lidar_scanner, flight_poses, attitudes)

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
                "seed": seed,
                "plane_size": PLANE_SIZE,
                "area_size": AREA_SIZE_M,
                "cell_size": CELL_SIZE_M,
                "cell_count": int(AREA_SIZE_M / CELL_SIZE_M),
                "grid_n": GRID_N,
                "ground_noise_cell_size": ground_noise_cell_size,
                "ground_noise_value": ground_noise_value,
                "altitude": altitude,
                "mask_path": mask_path,
                "mask_px": MASK_PX,
                "pix_size": AREA_SIZE_M / MASK_PX,
                "start_px": [int(start_px[0]), int(start_px[1])],
                "target_px": [int(target_px[0]), int(target_px[1])],
                "start": start_xy,
                "target": target_xy,
                "num_obstacles": num_obstacles,
                "num_rubble_piles": len(rubble_info),
                "num_barricades": len(barricade_info),
                "cleared_barricade_index": scene["cleared_barricade_index"],
                "rubble_info": rubble_info,
                "barricade_info": barricade_info,
                "scatter_info": scatter_info,
                "sensor": LIDAR_SENSOR_NAME,
                "sensor_params": LIDAR_SENSOR_PARAMS,
                "flight_poses": flight_poses,
                "attitude_jitter_deg": attitudes,
            }
            with open(os.path.join(scene_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            print(f"[scene {mask_idx} - {scene_idx}] sensor={LIDAR_SENSOR_NAME} obstacles={num_obstacles} "
                f"altitude={altitude:.1f}m start={start_xy} target={target_xy}")

        mask_idx += 1

    print("Done.")


if __name__ == "__main__":
    main()

"""Export a generated scene .blend file into a Gazebo (gz-sim) model directory.

Must be run with a *standalone* Blender (NOT blenderproc), since it uses the
built-in glTF exporter, e.g.:

    /home/dho/blender/blender-4.2.1-linux-x64/blender \\
        /path/to/scene_0000_0.blend --background --python \\
        gazebo/export_blend_to_gazebo.py -- \\
        --out /home/dho/work/ileri_otonom/shared/gz_ws/custom_models/rubble_scene_0000_0 \\
        --name rubble_scene_0000_0

Produces <out>/model.config, <out>/model.sdf, <out>/meshes/ground.<ext> and
<out>/meshes/obstacles.<ext>, ready to be referenced from a world file as
`model://<name>`. The `ground_base` mesh gets a cheap box collider sized to
its bounding box (instead of a per-triangle mesh collider); every other mesh
(rubble, barricades, scattered props) is exported as a visual-only obstacle
with no collider. Pass --format glb (default) or --format dae; glb (binary)
parses much faster in gz-sim than the verbose XML dae format once a scene
has more than a handful of objects.
"""
import argparse
import os
import sys

import bpy
import mathutils

MODEL_CONFIG_TEMPLATE = """<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>Procedurally generated obstacle terrain exported from Blender.</description>
</model>
"""

MODEL_SDF_HEADER = """<?xml version='1.0'?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="ground_collision">
        <pose>{box_cx} {box_cy} {box_cz} 0 0 0</pose>
        <geometry>
          <box><size>{box_sx} {box_sy} {box_sz}</size></box>
        </geometry>
      </collision>
      <visual name="ground_visual">
        <geometry>
          <mesh><uri>model://{name}/meshes/ground.{ext}</uri></mesh>
        </geometry>
      </visual>
"""

MODEL_SDF_OBSTACLES_VISUAL = """      <visual name="obstacles_visual">
        <geometry>
          <mesh><uri>model://{name}/meshes/obstacles.{ext}</uri></mesh>
        </geometry>
      </visual>
"""

MODEL_SDF_FOOTER = """    </link>
  </model>
</sdf>
"""


GROUND_PREFIXES = ("ground",)


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output model directory")
    parser.add_argument("--name", required=True, help="Gazebo model name")
    parser.add_argument("--format", choices=["glb", "dae"], default="glb",
                         help="mesh format to export (glb is much faster for gz-sim to parse)")
    return parser.parse_args(argv)


def export_selected(objs, filepath, fmt):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select_set(True)
    if fmt == "glb":
        # export_yup=False keeps Blender's native Z-up axes; otherwise the glTF
        # Y-up convention leaves everything rotated 90 degrees in gz-sim.
        bpy.ops.export_scene.gltf(filepath=filepath, export_format="GLB",
                                   use_selection=True, export_yup=False)
    else:
        bpy.ops.wm.collada_export(filepath=filepath, selected=True)


def merge_into_one_object(objs, name):
    """Join many small objects into a single mesh object to cut scene-graph/draw-call overhead."""
    if len(objs) <= 1:
        return objs
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[-1]
    bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active
    merged.name = name
    return [merged]


def world_bounding_box(objs):
    """Combined axis-aligned world-space bounding box (min, max) of objs."""
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for obj in objs:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                mins[i] = min(mins[i], world_corner[i])
                maxs[i] = max(maxs[i], world_corner[i])
    return mins, maxs


def main():
    args = parse_args()
    model_dir = args.out
    meshes_dir = os.path.join(model_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    # The generator scene mixes in lights/cameras; only meshes make up the terrain.
    all_mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    ground_objs = [obj for obj in all_mesh_objs if obj.name.lower().startswith(GROUND_PREFIXES)]
    # Everything that isn't the ground is a visual-only obstacle (no collider).
    obstacle_objs = [obj for obj in all_mesh_objs if not obj.name.lower().startswith(GROUND_PREFIXES)]
    if not ground_objs:
        raise RuntimeError("No ground mesh objects found (expected names starting with 'ground')")

    ground_objs = merge_into_one_object(ground_objs, "ground_merged")
    obstacle_objs = merge_into_one_object(obstacle_objs, "obstacles_merged")

    ext = args.format
    ground_path = os.path.join(meshes_dir, f"ground.{ext}")
    export_selected(ground_objs, ground_path, ext)

    box_min, box_max = world_bounding_box(ground_objs)
    box_center = [(box_min[i] + box_max[i]) / 2.0 for i in range(3)]
    box_size = [max(box_max[i] - box_min[i], 0.01) for i in range(3)]

    sdf = MODEL_SDF_HEADER.format(
        name=args.name, ext=ext,
        box_cx=box_center[0], box_cy=box_center[1], box_cz=box_center[2],
        box_sx=box_size[0], box_sy=box_size[1], box_sz=box_size[2],
    )
    if obstacle_objs:
        obstacles_path = os.path.join(meshes_dir, f"obstacles.{ext}")
        export_selected(obstacle_objs, obstacles_path, ext)
        sdf += MODEL_SDF_OBSTACLES_VISUAL.format(name=args.name, ext=ext)
    sdf += MODEL_SDF_FOOTER

    with open(os.path.join(model_dir, "model.config"), "w") as f:
        f.write(MODEL_CONFIG_TEMPLATE.format(name=args.name))
    with open(os.path.join(model_dir, "model.sdf"), "w") as f:
        f.write(sdf)

    print(f"Exported {len(ground_objs)} ground object(s) -> {ground_path}")
    if obstacle_objs:
        print(f"Exported {len(obstacle_objs)} obstacle/prop object(s) (visual-only) -> {obstacles_path}")
    print(f"Gazebo model ready at: {model_dir}")


if __name__ == "__main__":
    main()


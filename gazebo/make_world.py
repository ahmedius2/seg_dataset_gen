"""Generate a gz-sim world .sdf that drops an exported terrain model into the
scene alongside an ArduPilot-ready vehicle, positioned at the mask's start cell.

Usage (plain python, no bpy/blenderproc needed):

    python gazebo/make_world.py \\
        --terrain-model rubble_scene_0000_0 \\
        --out /home/dho/work/ileri_otonom/shared/gz_ws/custom_worlds/rubble_scene_0000_0.sdf \\
        --meta output/scene_0000/meta.json \\
        --vehicle-model iris_with_ardupilot

If --meta is omitted (e.g. SKIP_RENDER_AND_SCAN was on, so no meta.json was
written), the vehicle spawns at the world origin.
"""
import argparse
import json
import os

WORLD_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="{world_name}">
    <physics name="1ms" type="ignore">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"></plugin>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"></plugin>

    <scene>
      <ambient>1.0 1.0 1.0</ambient>
      <background>0.8 0.8 0.8</background>
      <sky></sky>
    </scene>

    <spherical_coordinates>
      <latitude_deg>-35.363262</latitude_deg>
      <longitude_deg>149.165237</longitude_deg>
      <elevation>584</elevation>
      <heading_deg>0</heading_deg>
      <surface_model>EARTH_WGS84</surface_model>
    </spherical_coordinates>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.8 0.8 0.8 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <include>
      <uri>model://{terrain_model}</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>

    <include>
      <uri>model://{vehicle_model}</uri>
      <pose degrees="true">{spawn_x} {spawn_y} {spawn_z} 0 0 90</pose>
    </include>

  </world>
</sdf>
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain-model", required=True, help="Gazebo model name of the exported terrain")
    parser.add_argument("--out", required=True, help="output world .sdf path")
    parser.add_argument("--meta", default=None, help="path to a scene meta.json (for start position)")
    parser.add_argument("--vehicle-model", default="iris_with_ardupilot")
    parser.add_argument("--world-name", default=None)
    parser.add_argument("--spawn-height", type=float, default=1.0,
                         help="metres above the terrain's z=0 plane to spawn the vehicle")
    return parser.parse_args()


def main():
    args = parse_args()
    spawn_x, spawn_y = 125.0, 0.0 # somewhere that is ground
    if args.meta:
        with open(args.meta) as f:
            meta = json.load(f)
        spawn_x, spawn_y = meta["start"]

    world_name = args.world_name or args.terrain_model
    world_sdf = WORLD_TEMPLATE.format(
        world_name=world_name,
        terrain_model=args.terrain_model,
        vehicle_model=args.vehicle_model,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        spawn_z=args.spawn_height,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(world_sdf)
    print(f"Wrote world file: {args.out}")
    print(f"Vehicle spawn: x={spawn_x} y={spawn_y} z={args.spawn_height}")


if __name__ == "__main__":
    main()

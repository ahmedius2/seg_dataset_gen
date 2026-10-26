import collections
import collections.abc

# Patch collections.MutableMapping for Python 3.10+ compatibility with DroneKit
collections.MutableMapping = collections.abc.MutableMapping

import hashlib
import random
import time
import math
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from dronekit import connect, VehicleMode
from pymavlink import mavutil

def generate_random_seed(string):
    """Generate a random seed from a given string for reproducibility."""
    digest = hashlib.sha256(string.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")

def load_spawn_position(scene_name):
    """Read the vehicle's world-frame spawn pose from the generated SDF."""
    world_filename = f"{scene_name}.sdf"
    search_dirs = [os.getcwd()]
    search_dirs.extend(os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(os.pathsep))
    search_dirs.append(os.path.expanduser("~/shared/custom_worlds"))

    world_path = next(
        (os.path.join(directory, world_filename)
         for directory in search_dirs
         if directory and os.path.isfile(os.path.join(directory, world_filename))),
        None,
    )
    if world_path is None:
        raise FileNotFoundError(
            f"Could not find {world_filename} in the current directory, GZ_SIM_RESOURCE_PATH, or ~/shared/custom_worlds"
        )

    root = ET.parse(world_path).getroot()
    for include in root.iter("include"):
        uri = include.findtext("uri", "").rstrip("/")
        if uri.rsplit("/", 1)[-1] != "iris_with_lidar":
            continue

        pose = include.findtext("pose")
        if pose is None:
            raise ValueError(f"Vehicle include in {world_path} has no pose")
        coordinates = pose.split()
        if len(coordinates) < 3:
            raise ValueError(f"Vehicle pose in {world_path} must contain x, y, and z")
        return {
            "x": float(coordinates[0]),
            "y": float(coordinates[1]),
            "z": float(coordinates[2]),
        }

    raise ValueError(f"No iris_with_lidar vehicle include found in {world_path}")

# Define output file and command
if len(sys.argv) < 3 or sys.argv[2] not in ("takeoff", "mission", "mission_norotate"):
    print("Usage: python collect_data.py <run_directory> <takeoff|mission|mission_norotate>")
    print("  takeoff : connect, arm, take off, and navigate to the scan start position, then exit")
    print("  mission : connect to the already-flying vehicle, run the scan pattern, RTL, then exit")
    print("  mission_norotate : connect to the already-flying vehicle, run the"
          " scan pattern without random rotations, RTL, then exit")
    sys.exit(1)

RUN_DIR = sys.argv[1]
PHASE = sys.argv[2]
print(f"Data collection run directory: {RUN_DIR}")
print(f"Phase: {PHASE}")
os.makedirs(RUN_DIR, exist_ok=True)

# generate the random seed number using the scene name environment variable, which is set by the calling script
SCENE_NAME = os.getenv('SCENE_NAME', 'scene_0000_0')
RANDOM_SEED = generate_random_seed(SCENE_NAME)

# --- 1. CONFIGURE SCANNING & SPAWN PARAMETERS ---
CONNECTION_STRING = 'tcp:127.0.0.1:14551'  # connect to mavproxy
RETRY_INTERVAL_SECONDS = 10

TARGET_SPEED = 10.0  # m/s
TARGET_REL_ALTITUDE = 35.0  # m above ground (Gazebo Z)

# The SDF pose is the stable ENU origin for ArduPilot's local NED frame.
SPAWN_POS = load_spawn_position(SCENE_NAME)
print(f"Vehicle Gazebo spawn position: {SPAWN_POS}")

# Define area bounds centered at (0,0) in Gazebo ENU
SCAN_AREA_SIZE = (150.0, 150.0)  # (width, height)
half_width = SCAN_AREA_SIZE[0] / 2.0
half_height = SCAN_AREA_SIZE[1] / 2.0

CORNERS = [
    (-half_width, -half_height),  # Bottom-Left
    (-half_width, half_height),   # Top-Left
    (half_width, half_height),    # Top-Right
    (half_width, -half_height)    # Bottom-Right
]

SWATH_WIDTH = 15.0  # Spacing between scan lines


def gazebo_to_ned(x, y, z, spawn=SPAWN_POS):
    """
    Convert absolute Gazebo ENU coordinates to local NED targets.
    """
    north = float(y) - spawn['y']
    east = float(x) - spawn['x']
    down = -(float(z) - spawn['z'])
    return north, east, down

def set_speed(vehicle, speed):
    """Sets vehicle ground speed in GUIDED mode."""
    print(f"Setting ground speed to: {speed} m/s")
    vehicle.groundspeed = speed

def connect_with_retry():
    """Retry the MAVLink connection until the vehicle sends a heartbeat."""
    while True:
        try:
            return connect(
                CONNECTION_STRING,
                wait_ready=False,
                heartbeat_timeout=RETRY_INTERVAL_SECONDS,
            )
        except Exception as error:
            print(f"Vehicle not available ({error}); retrying in {RETRY_INTERVAL_SECONDS} seconds.")
            time.sleep(RETRY_INTERVAL_SECONDS)

def goto_local_ned(vehicle, north, east, down):
    """Sends MAVLink position target command in LOCAL_NED frame."""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b000011110000, # Position-only mask
        north, east, down,
        0, 0, 0, # Velocities
        0, 0, 0, # Accelerations
        0, 0     # Yaw, Yaw_rate
    )
    vehicle.send_mavlink(msg)

def wait_to_reach_target(vehicle, target_n, target_e, target_d, tolerance=5.0):
    """Blocks until the drone is within 'tolerance' meters of the target."""
    while True:
        curr_n = vehicle.location.local_frame.north
        curr_e = vehicle.location.local_frame.east
        curr_d = vehicle.location.local_frame.down

        if curr_n is None or curr_e is None or curr_d is None:
            time.sleep(0.2)
            continue

        dist = math.sqrt((target_n - curr_n)**2 + (target_e - curr_e)**2 + (target_d - curr_d)**2)
        print(f" Distance to target: {dist:.2f} m  ", end="\r")

        if dist <= tolerance:
            print(f"\n Waypoint reached.")
            break

        goto_local_ned(vehicle, target_n, target_e, target_d)
        time.sleep(0.2)

def arm_and_takeoff_local(vehicle, start_x, start_y, target_alt):
    """Arms vehicle, takes off to target height, and navigates to start position."""
    print("Performing pre-arm checks...")
    while not vehicle.is_armable:
        print(f"Vehicle not armable yet; checking again in {RETRY_INTERVAL_SECONDS} seconds.")
        time.sleep(RETRY_INTERVAL_SECONDS)

    print("Setting mode to GUIDED...")
    while vehicle.mode.name != "GUIDED":
        try:
            vehicle.mode = VehicleMode("GUIDED")
        except Exception as error:
            print(f"Could not set GUIDED mode ({error}); retrying in {RETRY_INTERVAL_SECONDS} seconds.")
        if vehicle.mode.name != "GUIDED":
            time.sleep(RETRY_INTERVAL_SECONDS)

    print("Arming motors...")
    while not vehicle.armed:
        try:
            vehicle.armed = True
        except Exception as error:
            print(f"Could not arm vehicle ({error}); retrying in {RETRY_INTERVAL_SECONDS} seconds.")
        if not vehicle.armed:
            time.sleep(RETRY_INTERVAL_SECONDS)

    # Takeoff height is relative to spawn height
    takeoff_height = target_alt - SPAWN_POS['z']
    print(f"Taking off to target height relative to spawn: {takeoff_height}m (Abs Z: {target_alt}m)")
    highest_altitude = vehicle.location.global_relative_frame.alt
    last_progress_time = time.monotonic()
    last_command_time = 0.0
    takeoff_command_sent = False
    while True:
        curr_alt = vehicle.location.global_relative_frame.alt
        if curr_alt is not None and curr_alt >= takeoff_height * 0.95:
            print(f" Reached altitude: {curr_alt:.2f}m")
            break

        now = time.monotonic()
        if curr_alt is not None and (
            highest_altitude is None or curr_alt >= highest_altitude + 0.5
        ):
            highest_altitude = curr_alt
            last_progress_time = now

        stalled = now - last_progress_time >= RETRY_INTERVAL_SECONDS
        retry_interval_elapsed = now - last_command_time >= RETRY_INTERVAL_SECONDS
        if not takeoff_command_sent or (stalled and retry_interval_elapsed):
            if takeoff_command_sent:
                print(f"No altitude progress for {RETRY_INTERVAL_SECONDS} seconds; retrying takeoff.")
            try:
                vehicle.simple_takeoff(takeoff_height)
                takeoff_command_sent = True
            except Exception as error:
                print(f"Takeoff command failed ({error}); retrying in {RETRY_INTERVAL_SECONDS} seconds.")
            last_command_time = now

        time.sleep(1)

    print(f"Navigating to initial start point: Gazebo X={start_x}, Y={start_y}")
    n, e, d = gazebo_to_ned(start_x, start_y, target_alt)
    wait_to_reach_target(vehicle, n, e, d)

def rotate_point(x, y, cx, cy, angle_rad):
    """Rotates point (x, y) around (cx, cy) by angle_rad."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    nx = cx + cos_a * (x - cx) - sin_a * (y - cy)
    ny = cy + sin_a * (x - cx) + cos_a * (y - cy)
    return nx, ny

def generate_rotated_scan_path(corners, spacing, angle_degrees=0.0, pivot=None):
    # 1. Determine pivot point
    if pivot is None:
        cx = sum(c[0] for c in corners) / len(corners)
        cy = sum(c[1] for c in corners) / len(corners)
    else:
        cx, cy = pivot

    # 2. Get standard (unrotated) bounding box of the polygon
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 3. Generate standard vertical lawnmower path
    x_steps = np.arange(min_x, max_x + 1e-5, spacing)
    standard_waypoints = []

    for leg_count, current_x in enumerate(x_steps):
        if leg_count % 2 == 0:
            standard_waypoints.append((current_x, min_y))
            standard_waypoints.append((current_x, max_y))
        else:
            standard_waypoints.append((current_x, max_y))
            standard_waypoints.append((current_x, min_y))

    # 4. Rotate all generated waypoints at once
    angle_rad = math.radians(angle_degrees)
    rotated_waypoints = [
        rotate_point(x, y, cx, cy, angle_rad) for x, y in standard_waypoints
    ]

    # reversing is not necessary but it makes the first waypoint the
    # bottom-left corner of the scan area, which is more intuitive
    return list(reversed(rotated_waypoints))

# --- MAIN RUN SCRIPT ---
if __name__ == '__main__':
    print(f"Connecting to SITL on {CONNECTION_STRING} through mavproxy...")
    vehicle = connect_with_retry()

    try:
        if PHASE == "takeoff":
            # --- PHASE 1: Arm, take off, and navigate to the scan start position ---
            arm_and_takeoff_local(
                vehicle,
                SPAWN_POS['x'],
                SPAWN_POS['y'],
                SPAWN_POS['z'] + TARGET_REL_ALTITUDE,
            )
            print("\nTakeoff phase complete. Vehicle is holding at the scan start position.")
            print("You can now start rosbag recording, then run this script again with the 'mission' phase.")

        elif PHASE == "mission" or PHASE == "mission_norotate":
            # --- PHASE 2: Run the scan pattern and RTL ---
            # Generate scan plan waypoints
            if PHASE == "mission_norotate":
                random_rotate_angle = 0.0
                print(f"Random rotation angle: {random_rotate_angle:.1f}° (No rotation)")
            else:
                random_rotate_angle = random.Random(RANDOM_SEED).uniform(0, 30)
                print(f"Random rotation angle: {random_rotate_angle:.1f}°")
            scan_waypoints = generate_rotated_scan_path(CORNERS, SWATH_WIDTH, random_rotate_angle)
            print(f"\nGenerated scan path with {len(scan_waypoints)} waypoints.")

            set_speed(vehicle, TARGET_SPEED)

            # Execute Lawnmower Grid
            for i, (wx, wy) in enumerate(scan_waypoints):
                # Vary scan altitude by up to 10 m around the target relative altitude.
                if PHASE == "mission":
                    new_altitude = random.Random(RANDOM_SEED + i).uniform(TARGET_REL_ALTITUDE - 10.0, TARGET_REL_ALTITUDE + 10.0)
                else:
                    new_altitude = TARGET_REL_ALTITUDE
                target_z = SPAWN_POS['z'] + new_altitude
                print(f"\n[Leg {i+1}/{len(scan_waypoints)}] Navigating to Gazebo (X: {wx:.1f}, Y: {wy:.1f}, Z: {target_z:.1f})")

                target_n, target_e, target_d = gazebo_to_ned(wx, wy, target_z)
                wait_to_reach_target(vehicle, target_n, target_e, target_d)

            print("\nScan pattern execution complete!")
            print("Returning to Launch (RTL)...")
            vehicle.mode = VehicleMode("RTL")

    finally:
        time.sleep(2)
        vehicle.close()
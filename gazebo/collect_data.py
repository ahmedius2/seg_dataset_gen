import collections
import collections.abc

# Patch collections.MutableMapping for Python 3.10+ compatibility with DroneKit
collections.MutableMapping = collections.abc.MutableMapping

import random
import time
import math
import os
import sys
import signal
import subprocess
import time
import numpy as np
from dronekit import connect, VehicleMode
from pymavlink import mavutil

def generate_random_seed(string):
    """Generate a random seed from a given string for reproducibility."""
    return hash(string) % (2**32)

# Define output file and command
if len(sys.argv) < 2:
    print("Usage: python collect_data.py <run_directory>")
    sys.exit(1)

RUN_DIR = sys.argv[1]
print(f"Data collection run directory: {RUN_DIR}")
os.makedirs(RUN_DIR, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "rosbag.log")
BAG_OUTPUT_PATH = os.path.join(RUN_DIR, "bag_recording")
ROSBAG_PROC = None  # Global variable to hold the rosbag process

# generate the random seed number using the scene name environment variable, which is set by the calling script
SCENE_NAME = os.getenv('SCENE_NAME', 'scene_0000_0')
RANDOM_SEED = generate_random_seed(SCENE_NAME)

# --- 1. CONFIGURE SCANNING & SPAWN PARAMETERS ---
CONNECTION_STRING = 'tcp:127.0.0.1:5762'  # SITL second connection

TARGET_SPEED = 5.0  # m/s
TARGET_ALTITUDE = 35.0  # m above ground (Gazebo Z)

# DRONE SPAWN POSITION IN GAZEBO (ENU)
# Used as offset because ArduPilot sets Local (0,0,0) at spawn location
SPAWN_POS = {'x': 150.0, 'y': 0.0, 'z': 1.0}

# Target start position in absolute Gazebo ENU coordinates (X, Y, Z) in meters
START_POS = {'x': 150.0, 'y': 0.0, 'z': TARGET_ALTITUDE} # The z here is the flying altitude for the scan pattern

# Define area bounds centered at (0,0) in Gazebo ENU
SCAN_AREA_SIZE = (100.0, 100.0)  # (width, height)
half_width = SCAN_AREA_SIZE[0] / 2.0
half_height = SCAN_AREA_SIZE[1] / 2.0

CORNERS = [
    (-half_width, -half_height),  # Bottom-Left
    (-half_width, half_height),   # Top-Left
    (half_width, half_height),    # Top-Right
    (half_width, -half_height)    # Bottom-Right
]

SWATH_WIDTH = 15.0  # Spacing between scan lines


def start_data_collection_rosbag():
    cmd = [
        "ros2", "bag", "record",
        "--use-sim-time",
        "-o", BAG_OUTPUT_PATH,
        "--topics",
        "/sim_lidar/pointcloud/downsampled",
        "/tf_throttled"
    ]
    # --- 1. START RECORDING ---
    with open(LOG_PATH, "w") as log_file:
        # start_new_session=True creates an isolated process group
        global ROSBAG_PROC
        ROSBAG_PROC = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        #check if the process started successfully
        time.sleep(5)  # Give it a moment to start
        if ROSBAG_PROC.poll() is not None:
            print(f"Failed to start rosbag recording. Check {LOG_PATH} for details.")
            sys.exit(1)

def stop_data_collection_rosbag():
    global ROSBAG_PROC
    if ROSBAG_PROC:
        os.killpg(os.getpgid(ROSBAG_PROC.pid), signal.SIGINT)
        try:
            ROSBAG_PROC.wait(timeout=10)
            print("Rosbag recording stopped successfully.")
        except subprocess.TimeoutExpired:
            print("Process timed out. Sending SIGKILL...")
            os.killpg(os.getpgid(ROSBAG_PROC.pid), signal.SIGKILL)

def gazebo_to_ned(x, y, z, spawn=SPAWN_POS):
    """
    Converts absolute Gazebo ENU coordinates (X, Y, Z) to ArduPilot
    Local NED coordinates relative to the vehicle's spawn location.
    """
    rel_x = x - spawn['x']
    rel_y = y - spawn['y']
    rel_z = z - spawn['z']

    north = float(rel_y)
    east = float(rel_x)
    down = float(-rel_z)
    return north, east, down

def set_speed(vehicle, speed):
    """Sets vehicle ground speed in GUIDED mode."""
    print(f"Setting ground speed to: {speed} m/s")
    vehicle.groundspeed = speed

def goto_local_ned(vehicle, north, east, down):
    """Sends MAVLink position target command in LOCAL_NED frame."""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111110000, # Position-only mask
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
        time.sleep(1)

    print("Setting mode to GUIDED...")
    vehicle.mode = VehicleMode("GUIDED")
    time.sleep(1)

    print("Arming motors...")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(0.5)

    # Takeoff height is relative to spawn height
    takeoff_height = target_alt - SPAWN_POS['z']
    print(f"Taking off to target height relative to spawn: {takeoff_height}m (Abs Z: {target_alt}m)")
    vehicle.simple_takeoff(takeoff_height)

    while True:
        curr_alt = vehicle.location.global_relative_frame.alt
        if curr_alt >= takeoff_height * 0.95:
            print(f" Reached altitude: {curr_alt:.2f}m")
            break
        time.sleep(0.5)

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

    return rotated_waypoints

# --- MAIN RUN SCRIPT ---
if __name__ == '__main__':
    print(f"Connecting to SITL on {CONNECTION_STRING}...")
    vehicle = connect(CONNECTION_STRING, wait_ready=True)

    # Generate scan plan waypoints
    random_rotate_angle = random.Random(RANDOM_SEED).uniform(0, 30)
    print(f"Random rotation angle: {random_rotate_angle:.1f}°")
    scan_waypoints = generate_rotated_scan_path(CORNERS, SWATH_WIDTH, random_rotate_angle)
    print(f"\nGenerated scan path with {len(scan_waypoints)} waypoints.")

    try:
        set_speed(vehicle, TARGET_SPEED)

        # Arm and fly to designated start position (X, Y, Z)
        arm_and_takeoff_local(vehicle, START_POS['x'], START_POS['y'], START_POS['z'])

        start_data_collection_rosbag()

        # Execute Lawnmower Grid
        for i, (wx, wy) in enumerate(scan_waypoints):
            # choose a new altitude randomly between TARGET_ALTITUDE and TARGET_ALTITUDE + 5.0 meters for each waypoint
            new_altitude = random.Random(RANDOM_SEED + i).uniform(TARGET_ALTITUDE - 10.0, TARGET_ALTITUDE + 10.0)
            print(f"\n[Leg {i+1}/{len(scan_waypoints)}] Navigating to Gazebo (X: {wx:.1f}, Y: {wy:.1f}, Z: {new_altitude:.1f})")

            target_n, target_e, target_d = gazebo_to_ned(wx, wy, new_altitude)
            wait_to_reach_target(vehicle, target_n, target_e, target_d)

        stop_data_collection_rosbag()

        print("\nScan pattern execution complete!")
        print("Returning to Launch (RTL)...")
        vehicle.mode = VehicleMode("RTL")

    finally:
        time.sleep(2)
        vehicle.close()

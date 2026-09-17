#!/bin/bash

SCENE_NAME="${SCENE_NAME:-scene_0000_0}"

echo "Running simulation for $SCENE_NAME..."

# Enable job control (-m) so each background job runs in its own process group
set -m

# 1. Create a unique run directory based on the current timestamp
RUN_DIR="${HOME}/shared/runs/${SCENE_NAME}_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$RUN_DIR"
echo "Starting simulation run. Logs saving to: $RUN_DIR"

# Array to store Process IDs (PIDs)
PIDS=()

# Helper function to launch background commands safely
run_job() {
    local log_file="$1"
    shift
    
    # Launch command in background, redirecting stdout/stderr to tee and detaching stdin
    "$@" </dev/null > >(tee "$log_file") 2>&1 &
    
    # Capture the PID
    local pid=$!
    PIDS+=("$pid")
}

# 2. Cleanup function to send SIGINT to processes
cleanup() {
    echo -e "\nCaught signal! Stopping all background processes..."
    
    for pid in "${PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            # Send SIGINT (-2) to process group so ROS2/Gazebo write logs cleanly
            kill -INT -- -"$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null
        fi
    done
    
    echo "Waiting for processes to exit gracefully..."
    sleep 2
    
    # Force kill lingering processes
    for pid in "${PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
        fi
    done

    echo "All processes terminated cleanly."
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

CUR_PWD=$PWD
# 3. Launch background commands
pushd $RUN_DIR

run_job "$RUN_DIR/gz_sim.log" gz sim -v4 -r ${SCENE_NAME}.sdf
run_job "$RUN_DIR/sim_vehicle.log" sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --mavproxy-args="--daemon"
run_job "$RUN_DIR/simulation_bridge.log" ros2 launch copter_lidar_gzsim simulation_bridge.launch.py
run_job "$RUN_DIR/topic_throttle.log" ros2 run topic_tools throttle messages /tf 20.0 /tf_throttled

sleep 90 # wait for initialization
printf "#################################################\n"
printf "#################################################\n"
printf "#################################################\n"
python $CUR_PWD/collect_data.py $RUN_DIR  # this task will end by itself

popd
cleanup

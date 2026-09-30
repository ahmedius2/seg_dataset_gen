#!/bin/bash

# run with EXEC_MODE=test to do a testing session with dnn

EXEC_MODE="${EXEC_MODE:-dataset_gen}"  # or test
SCENE_NAME="${SCENE_NAME:-scene_0000_0}"
RUN_DIR="${HOME}/shared/simulation_data/${SCENE_NAME}"
MODEL_SDF_DIR=${ROS_WORKSPACE}/src/copter_lidar_gzsim/models/iris_with_lidar

export SCENE_NAME=${SCENE_NAME}

if [[ "${EXEC_MODE}" == "dataset_gen" ]]; then
    # check if directory is already created with bag recording
    if [[ -d "${RUN_DIR}/bag_recording" ]]; then
        echo "Skipping ${RUN_DIR}"
        exit 0
    fi
    LIDAR_HZ="2"
else # test
    rm -rf ${RUN_DIR}/*log ${RUN_DIR}/logs
    LIDAR_HZ="10"
fi

pushd $MODEL_SDF_DIR
rm -f model.sdf # remove the existing symbolic link
ln -s model_${LIDAR_HZ}hz.sdf model.sdf
popd

echo "Running simulation for $SCENE_NAME..."

# Enable job control (-m) so each background job runs in its own process group
set -m

# 1. Create a unique run directory based on the current timestamp
mkdir -p ${RUN_DIR}
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
    sleep 5

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

sleep 120  # wait for initialization
printf "Waited 120 seconds for initialization, starting to takeoff.\n"
python $CUR_PWD/collect_data.py $RUN_DIR takeoff
if [[ "${EXEC_MODE}" == "dataset_gen" ]]; then
    run_job "$RUN_DIR/rosbag.log" ros2 bag record --use-sim-time -o "$RUN_DIR/bag_recording" \
            --topics /sim_lidar/pointcloud/downsampled /tf_throttled
    RECORD="true"
else
    RECORD="false"
fi

output_dir="${HOME}/shared/dnn_dataset/${SCENE_NAME}"
mkdir -p $output_dir
run_job "$RUN_DIR/pc_transform.log" ros2 launch pc_transform_cpp pc_transform.launch.py \
        world_frame:=${SCENE_NAME} \
        use_sim_time:=true \
        output_dir:=${output_dir} \
        save_clouds:=${RECORD} \
        save_poses:=${RECORD}

if [[ "${EXEC_MODE}" == "dataset_gen" ]]; then
    python $CUR_PWD/collect_data.py $RUN_DIR mission
else # test
    run_job "$RUN_DIR/ros2_dnn_infer.log" python $CUR_PWD/../lidar/train_dnn/ros2_infer.py
    python $CUR_PWD/collect_data.py $RUN_DIR mission_norotate
fi

popd

# check if ${RUN_DIR}/bag_recording exists, if not, exit with error code 1
if [[ ${EXEC_MODE} == "dataset_gen" ]]; then
    if [ ! -d "${RUN_DIR}/bag_recording" ]; then
        echo "Error: ${RUN_DIR}/bag_recording does not exist. Simulation failed."
        exit 1
    else
        echo "Simulation completed successfully. Bag recording is available at: ${RUN_DIR}/bag_recording"
        exit 0
    fi
fi

cleanup

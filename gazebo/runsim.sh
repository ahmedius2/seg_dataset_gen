#!/bin/bash

# takes a exec mode argument:
# runsim.sh (dataset_gen | test | test_remote)

EXEC_MODE="${1:-dataset_gen}"  # or test or test_remote
SCENE_NAME="${SCENE_NAME:-scene_0012_0}"
RUN_DIR="${HOME}/shared/simulation_data/${SCENE_NAME}"
DATA_OUT_DIR="${HOME}/shared/dnn_dataset/${SCENE_NAME}"
MODEL_SDF_DIR=${ROS_WORKSPACE}/src/copter_lidar_gzsim/models/iris_with_lidar
IP_SDF="$HOME/shared/ardupilot_gazebo/models/iris_with_ardupilot/model.sdf"
LOCAL_IP="10.42.0.1" # if running SITL on another machine
REMOTE_IP="10.42.0.231" # if running SITL on another machine

export SCENE_NAME=${SCENE_NAME}

rm -rf ${RUN_DIR}/*log ${RUN_DIR}/logs

if [[ "${EXEC_MODE}" == "dataset_gen" && -d "${RUN_DIR}/bag_recording" ]]; then
    # check if directory is already created with bag recording
    echo "Skipping ${RUN_DIR}"
    exit 0
fi

# Set LiDAR frequency
LIDAR_HZ="2" #"10"
LIDAR_SDF=${MODEL_SDF_DIR}/model.sdf
sed -i "s|<update\_rate>.*</|<update\_rate>${LIDAR_HZ}</|" "${LIDAR_SDF}"

# 1. Create a unique run directory based on the current timestamp
mkdir -p ${RUN_DIR} ${DATA_OUT_DIR}

echo "Running simulation for $SCENE_NAME in exec mode $EXEC_MODE"
echo "Logs saving to: $RUN_DIR"

# Enable job control (-m) so each background job runs in its own process group
set -m

# Array to store Process IDs (PIDs)
PIDS=()

# Helper function to launch background commands safely
run_job() {
    local log_file="$1"
    shift

    # Launch command in background:
    # redirecting stdout/stderr to tee and detaching stdin
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

LH="127.0.0.1"
if [[ $EXEC_MODE == "test_remote" ]]; then
    # SITL should be started on the remote if not started already

    IP_SIM=$LOCAL_IP
    IP_OF_SITL=$REMOTE_IP

    run_job "${RUN_DIR}"/mavproxy.log mavproxy.py --master=tcp:${IP_OF_SITL}:5760 \
             --out=tcpin:${LH}:14551 --sitl=${IP_OF_SITL}:5501 --non-interactive
else
    IP_SIM=$LH
    IP_OF_SITL=$LH

    run_job "${RUN_DIR}"/sitl.log sim_vehicle.py \
            -v ArduCopter -f gazebo-iris --model JSON --no-rebuild \
            --mavproxy-args="--out=tcpin:127.0.0.1:14551 --non-interactive"
fi

sed -i "s|<fdm_addr>.*</|<fdm_addr>${IP_SIM}</|" "${IP_SDF}"
run_job "$RUN_DIR/gz_sim.log" gz sim -v4 -r ${SCENE_NAME}.sdf

run_job "$RUN_DIR/simulation_bridge.log" \
        ros2 launch copter_lidar_gzsim simulation_bridge.launch.py

sleep 30 # enough time to kick off

python $CUR_PWD/collect_data.py $RUN_DIR takeoff # blocking call

RECORD="false"
if [[ "${EXEC_MODE}" == "dataset_gen" ]]; then
    run_job "$RUN_DIR/topic_throttle.log" \
            ros2 run topic_tools throttle messages /tf 20.0 /tf_throttled

    run_job "$RUN_DIR/rosbag.log" ros2 bag record --use-sim-tim \
            -o "$RUN_DIR/bag_recording" \
            --topics /sim_lidar/pointcloud/downsampled /tf_throttled
    RECORD="true"
fi

if [[ $EXEC_MODE != "test_remote" ]]; then
    run_job "$RUN_DIR/pc_transform.log" \
            ros2 launch pc_transform_cpp pc_transform.launch.py \
            use_sim_time:=true \
            world_frame:=${SCENE_NAME} \
            output_dir:=${DATA_OUT_DIR} \
            save_clouds:=${RECORD} \
            save_poses:=${RECORD}

    if [[ $EXEC_MODE != "dataset_gen" ]]; then
        PT="$CUR_PWD/../lidar"
        # Pytorch run
        export DNN_BACKEND=torch DNN_WEIGHTS=$PT/lite_model/best.pt
        # Or, CPU run
        #ONNX_PT=$(realpath ${PT}/lite_model/onnx_export/bev_net.onnx)
        #export DNN_BACKEND=onnx DNN_ONNX_MODEL=$ONNX_PT
        run_job "$RUN_DIR/ros2_dnn_infer.log" python ${PT}/gnd_seg_dnn/ros2_infer.py
    fi
fi

if [[ "${EXEC_MODE}" == "dataset_gen" ]]; then
    python $CUR_PWD/collect_data.py $RUN_DIR mission
else
    python $CUR_PWD/collect_data.py $RUN_DIR mission_norotate
fi

popd

cleanup

# check if ${RUN_DIR}/bag_recording exists, if not, exit with error code 1
if [[ ${EXEC_MODE} == "dataset_gen" ]]; then
    if [ ! -d "${RUN_DIR}/bag_recording" ]; then
        echo "Error: ${RUN_DIR}/bag_recording does not exist. Simulation failed."
        exit 1
    else
        echo "Simulation completed successfully."
        echo "Bag recording is available at: ${RUN_DIR}/bag_recording"
        exit 0
    fi
fi

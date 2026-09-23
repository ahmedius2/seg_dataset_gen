#/bin/bash
set -u
shopt -s nullglob

stop_transformer() {
    if [[ -n "${transformer_pid:-}" ]] && kill -0 "$transformer_pid" 2>/dev/null; then
        echo "Stopping pc_transform, PID ${transformer_pid}"
        kill -INT "$transformer_pid" 2>/dev/null || true

        # Give ROS time to shut down cleanly.
        wait "$transformer_pid" 2>/dev/null || true
    fi

    transformer_pid=""
}

cleanup() {
    stop_transformer
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SIM_DATA_DIR=${HOME}/shared/simulation_data
for scene_dir in ${SIM_DATA_DIR}/*
do
    # Remove the trailing slash.
    #scene_dir="${scene_dir%/}"

    # Ignore directories that do not contain the expected bag.
    if [[ ! -d "$scene_dir/bag_recording" ]]; then
        echo "Skipping '$scene_dir': no bag_recording directory"
        continue
    fi

    scene_name="$(basename "$scene_dir" | cut -d '_' -f 1-3)"

    printf '\n========================================\n'
    printf 'Scene: %s\n' "$scene_name"
    printf 'Directory: %s\n' "$scene_dir"
    printf '========================================\n'

    output_dir="${HOME}/shared/dnn_dataset/${scene_name}"
    mkdir -p "$output_dir"

    echo "Starting pc_transform..."

    ros2 launch pc_transform_cpp pc_transform.launch.py
        world_frame:=world \
        use_sim_time:=true \
        output_dir:=${output_dir} \
        save_clouds:=true \
        save_poses:=true \
        > "${output_dir}/pc_transform.log" 2>&1 &

    transformer_pid=$!

    # Check that the process did not terminate immediately.
    sleep 2

    if ! kill -0 "$transformer_pid" 2>/dev/null; then
        echo "ERROR: pc_transform failed to start for ${scene_name}"
        cat "${output_dir}/pc_transform.log"
        transformer_pid=""
        continue
    fi

    echo "Playing bag..."

    ros2 bag play $scene_dir/bag_recording_fixed --clock -m "tf_throttled:=tf"

    bag_status=$?

    echo "Bag playback finished with status ${bag_status}"

    # Stop pc_transform before moving to the next scene.
    stop_transformer

    # Brief pause to allow ROS processes and files to close cleanly.
    sleep 2

    if [[ "$bag_status" -ne 0 ]]; then
        echo "WARNING: bag playback failed for ${scene_name}"
    fi
done

echo "All scenes processed."

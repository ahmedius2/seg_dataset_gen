#!/usr/bin/env bash
# Example commands for the LiDAR training, evaluation, export, and ROS 2 flow.
#
# Before running:
#   cd /path/to/seg_dataset_gen/lidar
#   source ./setup_env.sh
#
# Then use one command at a time:
#   ./testall.sh train configs/lite_static_04.yaml
#   ./testall.sh eval lite_static_04_yaml
#   ./testall.sh export lite_static_04_yaml
#   ./testall.sh ros2 lite_static_04_yaml onnx

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHONPATH:-}" || ":${PYTHONPATH}:" != *:"${SCRIPT_DIR}":* ]]; then
    echo "Source the repository environment first:" >&2
    echo "  source ${SCRIPT_DIR}/setup_env.sh" >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"

usage() {
    cat <<EOF
Usage: $0 train CONFIG.yaml
       $0 {eval|export} RUN_DIR
       $0 ros2 RUN_DIR [torch|onnx|tidl]

Commands:
  train   Train using the YAML configuration.
  eval    Evaluate the best checkpoint in RUN_DIR.
  export  Export the best checkpoint to RUN_DIR/onnx.
  ros2    Run inference with the selected backend (default: torch).
EOF
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

case "${1:-}" in
    train)
        require_file "${2:?Training YAML required}"
        exec "${PYTHON}" -m gnd_seg_dnn.train \
            --config "$2"
        ;;
    eval)
        require_file "${2:?Run directory required}/best.pt"
        exec "${PYTHON}" -m gnd_seg_dnn.evaluate \
            --run-dir "$2" \
            --out "$2/evaluation.json"
        ;;
    export)
        require_file "${2:?Run directory required}/best.pt"
        exec "${PYTHON}" -m gnd_seg_dnn.export_onnx \
            --run-dir "$2"
        ;;
    ros2)
        exec "${PYTHON}" -m gnd_seg_dnn.ros2_infer \
            --run-dir "${2:?Run directory required}" --backend "${3:-torch}"
        ;;
    *)
        usage
        exit 2
        ;;
esac

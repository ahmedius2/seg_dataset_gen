#!/usr/bin/env bash
# Example commands for the LiDAR training, evaluation, export, and ROS 2 flow.
#
# Before running:
#   cd /path/to/seg_dataset_gen/lidar
#   source ./setup_env.sh
#
# Then use one command at a time:
#   ./lidar/testall.sh train
#   RUN_DIR=./dnn_runs_static ./lidar/testall.sh eval
#   RUN_DIR=./dnn_runs_static ./lidar/testall.sh export
#   ONNX_MODEL=./exports/static/bev_net.onnx ./lidar/testall.sh ros2

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHONPATH:-}" || ":${PYTHONPATH}:" != *:"${SCRIPT_DIR}":* ]]; then
    echo "Source the repository environment first:" >&2
    echo "  source ${SCRIPT_DIR}/setup_env.sh" >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/shared/dnn_dataset}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/dnn_runs_static}"
EXPORT_DIR="${EXPORT_DIR:-${SCRIPT_DIR}/exports/static}"
CKPT="${CKPT:-${RUN_DIR}/best.pt}"
ONNX_MODEL="${ONNX_MODEL:-${EXPORT_DIR}/bev_net.onnx}"
MAX_PILLARS="${MAX_PILLARS:-4096}"

usage() {
    cat <<EOF
Usage: $0 {train|eval|export|ros2}

Commands:
  train   Train a static lite model using DATA_ROOT and RUN_DIR.
  eval    Evaluate CKPT on the validation split.
  export  Export CKPT to EXPORT_DIR, including the static PFN.
  ros2    Run ROS 2 inference using ONNX_MODEL.

Override paths with:
  DATA_ROOT, RUN_DIR, EXPORT_DIR, CKPT, ONNX_MODEL, MAX_PILLARS
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
        exec "${PYTHON}" -m gnd_seg_dnn.train \
            --lite \
            --vfe static \
            --cell-size 0.4 \
            --data "${DATA_ROOT}" \
            --out "${RUN_DIR}"
        ;;
    eval)
        require_file "${CKPT}"
        exec "${PYTHON}" -m gnd_seg_dnn.evaluate \
            --ckpt "${CKPT}" \
            --data "${DATA_ROOT}" \
            --out "${RUN_DIR}/evaluation.json"
        ;;
    export)
        require_file "${CKPT}"
        mkdir -p "${EXPORT_DIR}"
        exec "${PYTHON}" -m gnd_seg_dnn.export_onnx \
            --ckpt "${CKPT}" \
            --out "${EXPORT_DIR}" \
            --max-pillars "${MAX_PILLARS}"
        ;;
    ros2)
        require_file "${ONNX_MODEL}"
        export DNN_BACKEND=onnx
        export DNN_ONNX_MODEL="${ONNX_MODEL}"
        exec "${PYTHON}" -m gnd_seg_dnn.ros2_infer
        ;;
    *)
        usage
        exit 2
        ;;
esac

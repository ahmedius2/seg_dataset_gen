#!/usr/bin/env bash
# Source this file from the lidar directory to prepare the LiDAR environment.
#
# Usage:
#   source ./setup_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this script instead: source $0" >&2
    exit 1
fi

LIDAR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEG_DATASET_GEN_ROOT="$(cd -- "${LIDAR_ROOT}/.." && pwd)"

export SEG_DATASET_GEN_ROOT
export LIDAR_ROOT
export DNN_ROOT="${LIDAR_ROOT}"

case ":${PYTHONPATH:-}:" in
    *:"${LIDAR_ROOT}":*) ;;
    *) export PYTHONPATH="${LIDAR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

if ! command -v python >/dev/null 2>&1; then
    echo "python was not found on PATH; activate the intended virtual environment first" >&2
    return 1
fi

echo "Building the C++ voxelizer extension..."
(
    cd "${LIDAR_ROOT}" || exit
    python setup.py build_ext --inplace
) || {
    echo "C++ voxelizer build failed. Install pybind11 and a C++17 compiler." >&2
    return 1
}

echo "LiDAR environment ready:"
echo "  DNN_ROOT=${DNN_ROOT}"
echo "  PYTHONPATH includes ${LIDAR_ROOT}"

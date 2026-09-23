#!/usr/bin/env bash
# Export a generated .blend scene into a ready-to-fly Gazebo world for ArduPilot SITL.
#
# Usage:
#   ./gazebo/export_to_gazebo.sh <scene.blend> <model_name> [meta.json] [format]
#
# format is "glb" (default, fast to load in gz-sim) or "dae" (COLLADA).
#
# Example:
#   ./gazebo/export_to_gazebo.sh \
#       lidar/output/scenes/scene_0000_0.blend \
#       scene_0000_0
#
# Writes the model into $SHARED_WS/custom_models/<model_name> and the
# world into $SHARED_WS/custom_worlds/<model_name>.sdf, both already on the
# host path bind-mounted into the gz_ardupilot container as ~/shared/gz_ws.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_BIN="${BLENDER_BIN:-${HOME}/blender/blender-4.2.1-linux-x64/blender}"
SHARED_WS="${SHARED_WS:-${HOME}/work/ileri_otonom/shared}"
SCENE_NAME="${SCENE_NAME:-scene_0000_0}"
BLEND_FILE="$1"
META_JSON="${3:-}"
MESH_FORMAT="${4:-glb}"

MODEL_DIR="$SHARED_WS/custom_models/$SCENE_NAME"
WORLD_FILE="$SHARED_WS/custom_worlds/$SCENE_NAME.sdf"

"$BLENDER_BIN" "$BLEND_FILE" --background --python "$SCRIPT_DIR/export_blend_to_gazebo.py" -- \
  --out "$MODEL_DIR" --name "$SCENE_NAME" --format "$MESH_FORMAT"

META_ARGS=()
if [[ -n "$META_JSON" ]]; then
  META_ARGS=(--meta "$META_JSON")
fi
python3 "$SCRIPT_DIR/make_world.py" \
  --terrain-model "$SCENE_NAME" \
  --out "$WORLD_FILE" \
  "${META_ARGS[@]}"

#echo "Done. Inside the gz_ardupilot container run:"
#echo "  gz sim -v4 -r $SCENE_NAME.sdf"
#echo "  sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON"

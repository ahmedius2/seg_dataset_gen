#!/bin/bash

SCENES_DIR="${HOME}/work/ileri_otonom/shared/generated_scenes"
for f in "${SCENES_DIR}"/*.blend
do
    echo $f
    scene_name=$(basename "$f" .blend)
    echo "Exporting $scene_name to Gazebo..."
    export SCENE_NAME="$scene_name"
    ./export_to_gazebo.sh "$f" "$scene_name"
done

unset SCENE_NAME

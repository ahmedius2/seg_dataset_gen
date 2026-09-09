#!/bin/bash

for f in ${HOME}/work/ileri_otonom/seg_dataset_gen/lidar/output/scenes/*.blend; do
    scene_name=$(basename "$f" .blend)
    echo "Exporting $scene_name to Gazebo..."
    export SCENE_NAME="$scene_name"
    ./export_to_gazebo.sh "$f" "$scene_name"
done

unset SCENE_NAME
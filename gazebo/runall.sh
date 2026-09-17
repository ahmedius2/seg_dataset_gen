#!/bin/bash

SCENES_DIR="${HOME}/shared/seg_dataset_gen/lidar/output/scenes"

for f in ${SCENES_DIR}/*.blend
do
    sname=$(basename "$f" .blend)
    export SCENE_NAME=$sname
    ./runsim.sh
done

unset SCENE_NAME

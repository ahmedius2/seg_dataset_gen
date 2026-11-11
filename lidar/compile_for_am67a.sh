#!/bin/bash

# TIDL
pushd ../../edgeai-tidl-tools
source setup_env.sh am67a
popd

MODEL_DIR=statpillars_xyz_stats_04
rm -rf ${MODEL_DIR}/onnx
python -m gnd_seg_dnn.export_onnx --run-dir ${MODEL_DIR} \
        --single-pool
python3 -m gnd_seg_dnn.tidl_compile --run-dir ${MODEL_DIR} \
	--calib-dir /home/root/shared/dnn_dataset/scene_0000_0 \
	--debug-level 1 --bits 16
#        --pfn-deny-types MaxPool,Concat,Relu,ReduceMax,Expand,Constant,ConstantOfShape,Mul,Equal,Where
#        --pfn-deny-types MaxPool,Concat,Relu
#        --pfn-deny-names /Relu_1,/Relu_2,/Relu_3,/Relu_4,/Relu_5
	#--pfn-bits 8
#--skip-pfn

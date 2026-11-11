#!/bin/bash

# TIDL
pushd ../../edgeai-tidl-tools
source setup_env.sh am67a
popd

rm -rf single_z16/onnx
python -m gnd_seg_dnn.export_onnx --run-dir single_z16 \
        --single-pool
python3 -m gnd_seg_dnn.tidl_compile --run-dir single_z16 \
	--calib-dir /home/root/shared/dnn_dataset/scene_0000_0 \
	--debug-level 1 --bits 16
#        --pfn-deny-types MaxPool,Concat,Relu,ReduceMax,Expand,Constant,ConstantOfShape,Mul,Equal,Where
#        --pfn-deny-types MaxPool,Concat,Relu
#        --pfn-deny-names /Relu_1,/Relu_2,/Relu_3,/Relu_4,/Relu_5
	#--pfn-bits 8
#--skip-pfn

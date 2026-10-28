"""Build the C++ voxelizer in place:  python setup.py build_ext --inplace"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "gnd_seg_dnn._voxelizer",
    ["gnd_seg_dnn/csrc/voxelizer.cpp"],
    cxx_std=17,
    extra_compile_args=["-O3"],
    extra_link_args=["-pthread"],
)

setup(name="gnd_seg_dnn_voxelizer", ext_modules=[ext],
      cmdclass={"build_ext": build_ext}, packages=[])

import os
from glob import glob
from setuptools import find_packages, setup

package_name = "copter_lidar_gzsim"


# 1. Define the missing package_files helper function
def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


# Recursively collect all model files from models/
model_files = package_files("models")
data_files_dict = {}
for file_path in model_files:
    # Target directory in install/ share: share/copter_lidar_gzsim/models/...
    target_dir = os.path.join("share", package_name, os.path.dirname(file_path))
    if target_dir not in data_files_dict:
        data_files_dict[target_dir] = []
    data_files_dict[target_dir].append(file_path)

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
]

for target_dir, files in data_files_dict.items():
    data_files.append((target_dir, files))

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubuntu",
    maintainer_email="ubuntu@todo.todo",
    description="Drone LiDAR simulation package",
    license="Apache-2.0",
    extras_require={'test': ['pytest']},
    entry_points={
        "console_scripts": ['pc_transform = copter_lidar_gzsim.pc_transform:main'],
    },
)

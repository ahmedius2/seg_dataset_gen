import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

# Example command to get qos properties of topic
# ros2 topic info /sim_lidar/pointcloud/points -v

def generate_launch_description():
    pkg_share = get_package_share_directory("copter_lidar_gzsim")
    config_file_path = os.path.join(pkg_share, 'config', 'bridge_cfg.yaml')

    ros_gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="bridge_node",
        name="ros_gz_bridge",
        parameters=[
            {
                "use_sim_time": True,
                "qos_overrides./tf_static.publisher.durability": "transient_local",
                'qos_overrides./sim_lidar/pointcloud/points.publisher.reliability': 'best_effort',
                'qos_overrides./sim_lidar/pointcloud/points.publisher.durability': 'volatile',
                "config_file": config_file_path,
            }
        ],
        output="screen",
    )

    return LaunchDescription([ros_gz_bridge_node])

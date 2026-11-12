import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Example command to get qos properties of topic
# ros2 topic info /sim_lidar/pointcloud/points -v

def generate_launch_description():
    pkg_share = get_package_share_directory("copter_lidar_gzsim")
    config_file_path = os.path.join(pkg_share, 'config', 'bridge_cfg.yaml')

    # Args (override at launch time if needed)
    in_topic_arg = DeclareLaunchArgument(
        "in_topic", default_value="/sim_lidar/pointcloud/points",
        description="Raw point cloud topic published by ros_gz_bridge")
    out_topic_arg = DeclareLaunchArgument(
        "out_topic", default_value="/sim_lidar/pointcloud/compressed",
        description="Compressed point cloud topic published by republish")

    in_topic = LaunchConfiguration("in_topic")
    out_topic = LaunchConfiguration("out_topic")

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

    # republish: subscribe to raw cloud, publish compressed
    republish_node = Node(
        package="point_cloud_transport",
        executable="republish",
        name="pointcloud_republisher",
        parameters=[
            {
                "use_sim_time": True,
                "in_transport": "raw",
                "out_transport": "zstd",
            }
        ],
        remappings=[
            ("in", in_topic),
            ("out/zstd", out_topic),
        ],
        output="screen",
    )

    return LaunchDescription([
        in_topic_arg,
        out_topic_arg,
        ros_gz_bridge_node,
        republish_node,
    ])
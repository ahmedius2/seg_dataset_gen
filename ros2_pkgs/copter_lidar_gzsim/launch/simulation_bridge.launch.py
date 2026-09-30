import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

# Example command to get qos properties of topic
# ros2 topic info /sim_lidar/pointcloud/points -v

scene_name="scene_0000_0"

def generate_launch_description():
    pkg_share = get_package_share_directory("copter_lidar_gzsim")
    config_file_path = os.path.join(pkg_share, 'config', 'bridge_cfg.yaml')

    # 1. Define the Bridge as a ComposableNode
    ros_gz_bridge_node = ComposableNode(
        package="ros_gz_bridge",
        plugin="ros_gz_bridge::RosGzBridge",
        name="ros_gz_bridge",
        parameters=[
            {
                "use_sim_time": True,
                "qos_overrides./tf_static.publisher.durability": "transient_local",
                "config_file": config_file_path,
            }
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    # 2. Define the PCL VoxelGrid as a ComposableNode
    voxel_grid_node = ComposableNode(
        package="pcl_ros",
        plugin="pcl_ros::VoxelGrid",
        name="voxel_grid_filter",
        remappings=[
            ("input", "/sim_lidar/pointcloud/points"),
            ("output", "/sim_lidar/pointcloud/downsampled"),
        ],
        parameters=[{"leaf_size": 0.1},
                    {"filter_field_name": ""}],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    # 3. Wrap both ComposableNodes in a single Container (Enables Zero-Copy)
    pcl_and_bridge_container = ComposableNodeContainer(
        name="sensor_processing_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ros_gz_bridge_node,
            voxel_grid_node,
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            pcl_and_bridge_container,  # Pass the container here
            # RViz2 Node
#            Node(
#                package="rviz2",
#                executable="rviz2",
#                name="rviz2",
#                output="screen",
#                parameters=[{"use_sim_time": True}],
#            ),
        ]
    )

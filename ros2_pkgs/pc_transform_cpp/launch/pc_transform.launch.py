from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---- Declare every parameter as a launch argument ----
    declared_args = [
        DeclareLaunchArgument(
            'input_cloud_topic',
            default_value='/sim_lidar/pointcloud/downsampled',
            description='Input point cloud topic'),
        DeclareLaunchArgument(
            'output_cloud_topic',
            default_value='/sim_lidar/pointcloud/aligned',
            description='Output aligned point cloud topic'),
        DeclareLaunchArgument(
            'world_frame',
            default_value='world',
            description='World/reference frame'),
        DeclareLaunchArgument(
            'lidar_frame',
            default_value='iris_with_lidar/lidar_link/robosense_emx192',
            description='LiDAR sensor frame'),
        DeclareLaunchArgument(
            'drone_frame',
            default_value='iris_with_lidar',
            description='Drone base frame'),
        DeclareLaunchArgument(
            'ground_z',
            default_value='0.0',
            description='Ground plane z height'),
        DeclareLaunchArgument(
            'origin_mode',
            default_value='centroid',
            description='Origin mode (centroid, ...)'),
        DeclareLaunchArgument(
            'ground_band',
            default_value='1.0',
            description='Ground band thickness'),
        DeclareLaunchArgument(
            'mad_k',
            default_value='3.0',
            description='MAD multiplier for outlier rejection'),
        DeclareLaunchArgument(
            'min_ground_points',
            default_value='20',
            description='Minimum ground points'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation clock'),
        DeclareLaunchArgument(
            'save_clouds',
            default_value='False',
            description='Save clouds (dataset generation)'),
        DeclareLaunchArgument(
            'save_poses',
            default_value='False',
            description='Save poses (dataset generation)'),
        DeclareLaunchArgument(
            'output_dir',
            default_value='/tmp/dnn_dataset',
            description='Output directory for dataset'),
        DeclareLaunchArgument(
            'node_name',
            default_value='align_cloud_node',
            description='Node name'),
    ]

    # ---- Wire substitutions into the node parameters ----
    node = Node(
        package='pc_transform_cpp',
        executable='pc_transform_node',
        name=LaunchConfiguration('node_name'),
        output='screen',
        parameters=[{
            'input_cloud_topic':  LaunchConfiguration('input_cloud_topic'),
            'output_cloud_topic': LaunchConfiguration('output_cloud_topic'),
            'world_frame':        LaunchConfiguration('world_frame'),
            'lidar_frame':        LaunchConfiguration('lidar_frame'),
            'drone_frame':        LaunchConfiguration('drone_frame'),
            'ground_z':           LaunchConfiguration('ground_z'),
            'origin_mode':        LaunchConfiguration('origin_mode'),
            'ground_band':        LaunchConfiguration('ground_band'),
            'mad_k':              LaunchConfiguration('mad_k'),
            'min_ground_points':  LaunchConfiguration('min_ground_points'),
            'use_sim_time':       LaunchConfiguration('use_sim_time'),
            'save_clouds':        LaunchConfiguration('save_clouds'),
            'save_poses':         LaunchConfiguration('save_poses'),
            'output_dir':         LaunchConfiguration('output_dir'),
        }],
    )

    return LaunchDescription(declared_args + [node])

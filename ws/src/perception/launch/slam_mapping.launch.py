"""Opt-in async SLAM Toolbox mapping; no map_origin or privileged pose inputs."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('track', description='save with: ros2 run perception save_slam_map <track>'),
        Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
             output='screen', on_exit=Shutdown(reason='SLAM Toolbox stopped'),
             parameters=[PathJoinSubstitution([FindPackageShare('perception'), 'config', 'slam_mapping.yaml'])],
             remappings=[('pose', '/mapping/slam_pose')]),
        Node(package='perception', executable='slam_mapping_pose', name='slam_mapping_pose',
             output='screen', on_exit=Shutdown(reason='Mapping pose recorder stopped')),
    ])

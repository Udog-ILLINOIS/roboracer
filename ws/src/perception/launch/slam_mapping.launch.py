"""Opt-in async SLAM Toolbox mapping; no map_origin or privileged pose inputs."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    return LaunchDescription([
        # As mapping.launch.py: the bridge stamps with /clock time, so every node follows it
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        SetParameter(name='use_sim_time', value=sim_time),
        DeclareLaunchArgument('track', description='save with: ros2 run perception save_slam_map <track>'),
        Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
             output='screen', on_exit=Shutdown(reason='SLAM Toolbox stopped'),
             parameters=[PathJoinSubstitution([FindPackageShare('perception'), 'config', 'slam_mapping.yaml']),
                         {'use_sim_time': sim_time}],  # after the yaml, so it wins
             remappings=[('pose', '/mapping/slam_pose')]),
        Node(package='perception', executable='slam_mapping_pose', name='slam_mapping_pose',
             output='screen', on_exit=Shutdown(reason='Mapping pose recorder stopped')),
    ])

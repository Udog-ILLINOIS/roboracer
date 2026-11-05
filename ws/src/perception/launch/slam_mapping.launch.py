"""Opt-in async SLAM Toolbox mapping; no map_origin or privileged pose inputs."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        # same clock as the bridge and the rest of the stack (map_track.sh passes USE_SIM_TIME)
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        SetParameter(name='use_sim_time', value=ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)),
        DeclareLaunchArgument('track', description='save with: ros2 run perception save_slam_map <track>'),
        Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
             output='screen', on_exit=Shutdown(reason='SLAM Toolbox stopped'),
             parameters=[PathJoinSubstitution([FindPackageShare('perception'), 'config', 'slam_mapping.yaml'])],
             remappings=[('pose', '/mapping/slam_pose')]),
        Node(package='perception', executable='slam_mapping_pose', name='slam_mapping_pose',
             output='screen', on_exit=Shutdown(reason='Mapping pose recorder stopped')),
    ])

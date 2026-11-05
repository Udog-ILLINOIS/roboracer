"""ros2 launch perception ekf.launch.py

Experimental IMU velocity estimate, not used by the race pipeline (sim only: reads raw
/autodrive/roboracer_1/* topics). Run alongside the bridge:

  ekf_filter_node_accl (robot_localization):  sim IMU -> /odometry/filtered_accl
  corrector:  /odometry/filtered_accl + wheel encoders -> /odometry/corrected (drift-corrected vx),
              plus /odometry/velocity_error against sim ground truth
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ekf = Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node_accl',
               output='screen',
               parameters=[PathJoinSubstitution([FindPackageShare('perception'), 'config', 'ekf_config_accl.yaml'])],
               remappings=[('odometry/filtered', 'odometry/filtered_accl')])
    corrector = Node(package='perception', executable='corrector', name='ekf_corrector', output='screen')
    return LaunchDescription([
        # the bridge's clock, like the other launch files (sensor stamps are on /clock)
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        SetParameter(name='use_sim_time', value=ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)),
        ekf, corrector])

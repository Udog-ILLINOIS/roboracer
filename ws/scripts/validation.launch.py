"""Validation-only scope around the existing race launch; no privileged localizer hints."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    defaults = {'track': 'test_track_v2', 'localizer': 'mcl', 'speed_scale': '0.3', 'use_sim_time': 'true'}
    return LaunchDescription([
        *[DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()],
        GroupAction([
            SetRemap(src='/ground_truth/odom', dst='/validation/disabled_ground_truth'),
            SetRemap(src='/autodrive/roboracer_1/sim_reset', dst='/validation/disabled_reset_hint'),
            IncludeLaunchDescription(
                PathJoinSubstitution([FindPackageShare('control'), 'launch', 'race.launch.py']),
                launch_arguments={k: LaunchConfiguration(k) for k in defaults}.items()),
        ]),
    ])

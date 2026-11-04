"""Real async mapper and native serialization on synthetic scans, isolated from AutoDRIVE.

Produces a temporary room map, not a validation of track mapping/loop closure.
"""
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from PIL import Image

from perception.save_slam_map import SaveSlamMap, validate_artifacts, export_snapshot


def test_async_mapper_native_save(tmp_path):
    domain = '66'  # Never publish these synthetic sensors in the live stack's domain.
    env = dict(os.environ, ROS_DOMAIN_ID=domain)
    with (tmp_path / 'launch.log').open('w') as log:
        process = subprocess.Popen(['ros2', 'launch', 'perception', 'slam_mapping.launch.py', 'track:=synthetic',
                                     'use_sim_time:=false'],  # the driver below stamps with wall time, no /clock,
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rclpy.init(domain_id=int(domain))
        driver = rclpy.create_node('synthetic_mapping_sensors')
        saver = None
        try:
            scan_pub = driver.create_publisher(LaserScan, '/scan', 10)
            odom_pub = driver.create_publisher(Odometry, '/odom', 10)
            static, dynamic = StaticTransformBroadcaster(driver), TransformBroadcaster(driver)
            mount = TransformStamped()
            mount.header.frame_id, mount.child_frame_id = 'base_link', 'laser'
            mount.header.stamp = driver.get_clock().now().to_msg()
            mount.transform.translation.x = .2733
            mount.transform.rotation.w = 1.
            static.sendTransform(mount)
            end = time.monotonic() + 10
            while scan_pub.get_subscription_count() == 0 and time.monotonic() < end:
                rclpy.spin_once(driver, timeout_sec=.1)
                assert process.poll() is None, (tmp_path / 'launch.log').read_text()
            assert scan_pub.get_subscription_count() > 0
            # Drive synthetically along a room's centre: ray intersections with four walls.
            angles = -math.radians(135) + np.arange(271) * math.radians(1)
            dx, dy = np.cos(angles), np.sin(angles)
            for i in range(45):
                stamp = driver.get_clock().now().to_msg()
                x = min(i, 35) * .06
                tf = TransformStamped()
                tf.header.frame_id, tf.child_frame_id, tf.header.stamp = 'odom', 'base_link', stamp
                tf.transform.translation.x, tf.transform.rotation.w = float(x), 1.
                dynamic.sendTransform(tf)
                odom = Odometry()
                odom.header, odom.child_frame_id = tf.header, 'base_link'
                odom.pose.pose.position.x, odom.pose.pose.orientation.w = float(x), 1.
                odom_pub.publish(odom)
                scan = LaserScan()
                scan.header.frame_id, scan.header.stamp = 'laser', stamp
                scan.angle_min, scan.angle_increment = float(angles[0]), math.radians(1)
                scan.angle_max = float(angles[-1])
                scan.range_min, scan.range_max, scan.scan_time = .15, 10., .12
                with np.errstate(divide='ignore'):
                    tx = np.where(dx > 0, (5 - x - .2733) / dx, (-5 - x - .2733) / dx)
                    ty = np.where(dy > 0, 5 / dy, -5 / dy)
                scan.ranges = np.minimum(np.minimum(tx, ty), 10).astype(float).tolist()
                scan_pub.publish(scan)
                rclpy.spin_once(driver, timeout_sec=.02)
                time.sleep(.12)
            saver = SaveSlamMap()
            saver.timeout = 10.
            output = tmp_path / 'synthetic'
            saver.save(output)
            validate_artifacts(output)
            with Image.open(output / 'map.pgm') as image:
                pixels = np.asarray(image)
                assert image.width > 20 and image.height > 20
                assert (pixels == 0).any() and (pixels == 254).any()
            assert (output / 'map.posegraph').stat().st_size > 100
            assert (output / 'map.data').stat().st_size > 100
            # Independent re-export of this exact snapshot must reproduce map pixels and poses.
            second = tmp_path / 'independent_export'
            export_snapshot(str(output / 'map'), second)
            assert (output / 'map.pgm').read_bytes() == (second / 'map.pgm').read_bytes()
            assert np.allclose(np.loadtxt(output / 'trajectory.csv', delimiter=','),
                               np.loadtxt(second / 'trajectory.csv', delimiter=','))
            assert (output / 'trajectory_online.csv').is_file()
            assert (output / 'mapping_diagnostics.json').is_file()
            try:
                export_snapshot(str(output / 'map'), second)
                assert False, 'export must refuse an existing directory'
            except subprocess.CalledProcessError:
                pass
            nodes = driver.get_node_names()
            assert 'slam_toolbox' in nodes and 'slam_mapping_pose' in nodes
            assert 'cartographer_node' not in nodes
            subs = driver.get_subscriber_names_and_types_by_node('slam_toolbox', '/')
            assert not any('ground_truth' in topic or 'autodrive' in topic for topic, _ in subs)
            print(f'Synthetic native graph and occupancy map generated/inspected: {output}')
        finally:
            if saver:
                saver.destroy_node()
            driver.destroy_node()
            rclpy.try_shutdown()
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass

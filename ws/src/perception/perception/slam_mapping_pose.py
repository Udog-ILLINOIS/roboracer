"""Mapping-only native pose relay and online trajectory; never broadcasts TF."""
import math
import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


def valid_pose(msg):
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
    return (msg.header.frame_id == 'map' and all(map(math.isfinite, values))
            and abs(math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w) - 1) <= 0.1
            and (msg.header.stamp.sec > 0 or msg.header.stamp.nanosec > 0))


class SlamMappingPose(Node):
    def __init__(self):
        super().__init__('slam_mapping_pose')
        self.path = Path()
        self.path.header.frame_id = 'map'
        self.stats = {'scan_count': 0, 'pose_count': 0, 'max_scan_age_s': None,
                      'max_pose_age_s': None, 'pose_age_over_0_5_count': 0,
                      'max_scan_stamp_gap_s': None}
        self.last_scan_stamp = None
        self.pose_pub = self.create_publisher(PoseStamped, '/tracked_pose', 10)
        self.path_pub = self.create_publisher(Path, '/mapping/trajectory',
                                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(PoseWithCovarianceStamped, '/mapping/slam_pose', self.pose_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.diagnostics_pub = self.create_publisher(String, '/mapping/diagnostics',
                                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(0.5, self.publish_path)

    def pose_cb(self, msg):
        if not valid_pose(msg):
            return
        pose = PoseStamped(header=msg.header, pose=msg.pose.pose)
        # Reject duplicate/backwards stamps; preserve acquisition time and mapping direction.
        t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if self.path.poses:
            last = self.path.poses[-1].header.stamp
            if t <= last.sec * 10**9 + last.nanosec:
                return
        self.path.header = msg.header
        self.path.poses.append(pose)
        self.stats['pose_count'] += 1
        age = self.get_clock().now().nanoseconds / 1e9 - t / 1e9
        self.stats['max_pose_age_s'] = max(self.stats['max_pose_age_s'] or 0, age)
        self.stats['pose_age_over_0_5_count'] += int(age > .5)
        self.pose_pub.publish(pose)

    def scan_cb(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = self.get_clock().now().nanoseconds / 1e9 - stamp
        self.stats['scan_count'] += 1
        self.stats['max_scan_age_s'] = max(self.stats['max_scan_age_s'] or 0, age)
        if self.last_scan_stamp is not None:
            self.stats['max_scan_stamp_gap_s'] = max(self.stats['max_scan_stamp_gap_s'] or 0,
                                                    stamp - self.last_scan_stamp)
        self.last_scan_stamp = stamp

    def publish_path(self):
        self.diagnostics_pub.publish(String(data=json.dumps(self.stats)))
        if self.path.poses:
            self.path_pub.publish(self.path)


def main():
    rclpy.init()
    node = SlamMappingPose()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

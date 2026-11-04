"""Save a stopped SLAM Toolbox mapping session, without overwriting a track.

The installed Humble SerializePoseGraph API takes a prefix, writes .posegraph and
.data, and reports RESULT_SUCCESS=0. Occupancy grids retain Cartographer's map
encoding; final corrected graph poses supply the planner's direction/length checks; online poses are retained separately.
"""
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.utilities import remove_ros_args
from nav_msgs.msg import OccupancyGrid, Path as RosPath
from std_msgs.msg import String
from slam_toolbox.srv import SerializePoseGraph

from perception.tracks import tracks_dir
from perception.mapping_diagnostics import diagnose


def destination(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('track must be a simple directory name')
    return Path(tracks_dir()) / name


def write_grid(out, grid):
    w, h, res = grid.info.width, grid.info.height, grid.info.resolution
    q, p = grid.info.origin.orientation, grid.info.origin.position
    values = (res, p.x, p.y, q.x, q.y, q.z, q.w)
    if (grid.header.frame_id != 'map' or not all(map(math.isfinite, values))
            or not math.isclose(res, .05, abs_tol=1e-6) or w < 1 or h < 1
            or len(grid.data) != w*h or abs(q.z) > 1e-6 or abs(abs(q.w) - 1) > 1e-6
            or abs(q.x) > 1e-6 or abs(q.y) > 1e-6):
        raise RuntimeError('invalid map: expected map frame, 0.05 m resolution and unrotated finite origin')
    occ = np.asarray(grid.data, dtype=np.int16).reshape(h, w)[::-1]
    if not np.isin(occ, np.arange(-1, 101)).all() or not (occ == 0).any() or not (occ >= 65).any():
        raise RuntimeError('map has invalid occupancy or no free/occupied cells')
    pixels = np.full((h, w), 205, dtype=np.uint8)
    pixels[(occ >= 0) & (occ <= 25)] = 254
    pixels[occ >= 65] = 0
    (out / 'map.pgm').write_bytes(f'P5\n{w} {h}\n255\n'.encode() + pixels.tobytes())
    (out / 'map.yaml').write_text(
        f'image: map.pgm\nmode: trinary\nresolution: {res:.6f}\n'
        f'origin: [{p.x:.6f}, {p.y:.6f}, 0.0]\nnegate: 0\n'
        'occupied_thresh: 0.65\nfree_thresh: 0.196\n')


def write_trajectory(out, path):
    if path.header.frame_id != 'map' or len(path.poses) < 11:
        raise RuntimeError('missing mapping trajectory (need at least 11 map-frame poses)')
    rows, last = ['# t,x_m,y_m,yaw_rad'], -1
    for pose in path.poses:
        p, q = pose.pose.position, pose.pose.orientation
        t = pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9
        values = (t, p.x, p.y, q.x, q.y, q.z, q.w)
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if pose.header.frame_id != 'map' or not all(map(math.isfinite, values)) or abs(norm - 1) > .1 or t <= 0 or t <= last:
            raise RuntimeError('invalid/nonmonotonic mapping trajectory')
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        rows.append(f'{t:.6f},{p.x:.4f},{p.y:.4f},{yaw:.5f}')
        last = t
    (out / 'trajectory.csv').write_text('\n'.join(rows) + '\n')


def validate_artifacts(out):
    for name in ('map.pgm', 'map.yaml', 'map.posegraph', 'map.data', 'trajectory.csv'):
        if not (out / name).is_file() or (out / name).stat().st_size == 0:
            raise RuntimeError(f'missing/empty saved artifact: {name}')


def export_snapshot(prefix, directory):
    from ament_index_python.packages import get_package_prefix
    executable = Path(get_package_prefix('slam_graph_tools')) / 'lib/slam_graph_tools/export_slam_graph'
    subprocess.run([str(executable), prefix, str(directory)],
                   check=True, timeout=60)


def save_bundle(out, grid, trajectory, serialize, extract=None, runtime=None):
    """Stage on the same filesystem; reserve destination and refuse existing paths.

    Failure keeps the private staging directory for late service writes/inspection.
    Retrying creates a new stage, never overwrites a partial or validated track.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(out):
        raise FileExistsError(f'{out} exists; choose a new track name')
    stage = Path(tempfile.mkdtemp(prefix=f'.{out.name}-saving-', dir=out.parent))
    try:
        write_grid(stage, grid)
        write_trajectory(stage, trajectory)
        (stage / 'trajectory.csv').rename(stage / 'trajectory_online.csv')
        serialize(str(stage / 'map'))
        for name in ('map.posegraph', 'map.data'):
            if not (stage / name).is_file() or (stage / name).stat().st_size == 0:
                raise RuntimeError(f'missing/empty saved artifact: {name}')
        extract = extract or export_snapshot
        extract(str(stage / 'map'), stage / 'final_graph')
        # Both products derive from the same immutable snapshot, not a cached live grid
        # paired with a separately optimized historical pose stream.
        for name in ('map.pgm', 'map.yaml', 'trajectory.csv', 'odometry.csv', 'graph_metadata.json'):
            shutil.copyfile(stage / 'final_graph' / name, stage / name)
        report = diagnose(stage / 'map.yaml', stage / 'trajectory_online.csv',
                          stage / 'trajectory.csv', stage / 'odometry.csv')
        if (report['final']['motion']['samples'] < 11
                or report['online_to_final']['unmatched_online']):
            raise RuntimeError('serialized graph does not cover this mapping trajectory')
        report['runtime'] = runtime
        (stage / 'mapping_diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        validate_artifacts(stage)
        (stage / 'mapping.json').write_text(json.dumps({
            'backend': 'slam_toolbox', 'trajectory': 'corrected poses from serialized graph; map rasterized from same snapshot',
            'graph_prefix': 'map', 'resolution_m': .05}, indent=2) + '\n')
        os.chmod(stage, 0o755)  # mkdtemp makes it 0700; the track dir is read from the host too
        # mkdir is exclusive, including a competing saver/validated destination.
        out.mkdir()
        try:
            os.replace(stage, out)  # replace only the empty directory we just reserved
        except BaseException:
            out.rmdir()
            raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise RuntimeError(f'{error}; staging artifacts retained at {stage}') from error


class SaveSlamMap(Node):
    def __init__(self):
        super().__init__('save_slam_map')
        self.timeout = self.declare_parameter('timeout', 30.0).value
        self.grid = self.trajectory = None
        self.map_rx = 0.0
        self.runtime = None
        self.create_subscription(OccupancyGrid, '/map', self.map_cb,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(RosPath, '/mapping/trajectory', self.path_cb,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, '/mapping/diagnostics', self.diagnostics_cb,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def diagnostics_cb(self, msg):
        self.runtime = json.loads(msg.data)

    def map_cb(self, msg):
        self.grid, self.map_rx = msg, time.monotonic()

    def path_cb(self, msg):
        self.trajectory = msg

    def serialize(self, prefix):
        client = self.create_client(SerializePoseGraph, '/slam_toolbox/serialize_map')
        try:
            if not client.wait_for_service(timeout_sec=self.timeout):
                raise RuntimeError('SLAM Toolbox serialize service unavailable')
            future = client.call_async(SerializePoseGraph.Request(filename=prefix))
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout)
            if not future.done() or future.result() is None:
                raise RuntimeError('SLAM Toolbox serialization timed out')
            if future.result().result != SerializePoseGraph.Response.RESULT_SUCCESS:
                raise RuntimeError(f'SLAM Toolbox serialization failed: {future.result().result}')
        finally:
            self.destroy_client(client)

    def save(self, out):
        # Car must be stopped. Wait for a periodic map after this saver started,
        # rather than treating the first transient-local cached grid as current.
        after, end = time.monotonic(), time.monotonic() + self.timeout
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=.1)
            if (self.grid is not None and self.map_rx > after + 2.0
                    and self.trajectory is not None and len(self.trajectory.poses) >= 11):
                save_bundle(out, self.grid, self.trajectory, self.serialize, runtime=self.runtime)
                return
        raise RuntimeError('no fresh /map or mapping trajectory; keep mapping running and retry')


def main():
    args = remove_ros_args(sys.argv)[1:]
    if len(args) != 1:
        print('usage: ros2 run perception save_slam_map <new track name>')
        return 2
    node = None
    try:
        out = destination(args[0])
        if os.path.lexists(out):
            raise FileExistsError(f'{out} exists; choose a new track name')
        rclpy.init()
        node = SaveSlamMap()
        node.save(out)
        node.get_logger().info(f'saved SLAM Toolbox map and graph to {out}')
        return 0
    except (KeyboardInterrupt, ExternalShutdownException):
        return 130
    except (RuntimeError, ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        if node:
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())

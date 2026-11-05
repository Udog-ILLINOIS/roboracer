#!/usr/bin/env python3
"""Run inside a sourced ROS container; AutoDRIVE and its bridge must already be running.

Defaults: test_track_v2, MCL, speed scale 0.3, two laps. See validation.md.
The evaluator publishes only lifecycle reset requests, never driving commands or pose hints.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

NS = '/autodrive/roboracer_1/'
BASE_NODES = {'fast_bridge', 'adapter', 'wheel_odom'}
SENSORS = {'/scan', '/odom', '/sensors/imu/raw'}
BLOCKED = {'pure_pursuit_node', 'reactive_node', 'wall_follow_node', 'disparity_gap_node',
           'teleop_keyboard', 'cartographer_node', 'localization_gate', 'mcl', 'obstacle_spliner',
           'slam_toolbox', 'slam_mapping_pose', 'map_node'}
DISABLED = {'/validation/disabled_ground_truth', '/validation/disabled_reset_hint'}


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', default='test_track_v2')
    parser.add_argument('--localizer', choices=['mcl', 'cartographer'], default='mcl')
    parser.add_argument('--speed-scale', type=positive, default=0.3)
    parser.add_argument('--laps', type=int, default=2)
    parser.add_argument('--timeout', type=positive, default=180.0, help='total seconds including setup')
    parser.add_argument('--startup-timeout', type=positive, default=30.0)
    parser.add_argument('--tracks-dir', type=Path,
                        default=Path(os.environ.get('TRACKS_DIR', '~/ws/tracks')).expanduser())
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'log/validation')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.track):
        parser.error('track must be a simple directory name')
    if args.laps < 1:
        parser.error('laps must be positive')
    if args.speed_scale > 1:
        parser.error('speed scale must be <= 1')
    args.tracks_dir = args.tracks_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.results_dir == args.tracks_dir or args.tracks_dir in args.results_dir.parents:
        parser.error('results must be outside tracks-dir')
    return args


def sim_time():
    """The stack's clock choice, as common.sh: the bridge's /clock unless USE_SIM_TIME=false."""
    return os.environ.get('USE_SIM_TIME', 'true').lower() != 'false'


class Failure(RuntimeError):
    pass


class Observations:
    """Evaluation state only. No estimates are manufactured from missing telemetry."""
    def __init__(self, sink, clock=time.monotonic):
        self.sink, self.clock = sink, clock
        self.seen, self.values, self.baseline = {}, {}, {}
        self.counter_error = None
        self.pose_samples = 0

    def record(self, key, value):
        now = self.clock()
        if key in ('lap_count', 'collision_count') and key in self.values and value < self.values[key]:
            self.counter_error = key + ' decreased (reset or telemetry discontinuity)'
        self.seen[key], self.values[key] = now, value
        self.sink.write(json.dumps({'monotonic_s': now, 'topic': key, 'value': value}, allow_nan=False) + '\n')
        self.sink.flush()

    def fresh(self, key, age=2.0):
        return key in self.seen and self.clock() - self.seen[key] <= age

    def delta(self, key):
        if key not in self.baseline or key not in self.values:
            return None
        return self.values[key] - self.baseline[key]

    def metrics(self):
        return {'completed_laps': self.delta('lap_count'), 'collisions': self.delta('collision_count'),
                'pose_samples': self.pose_samples,
                'telemetry_available': {k: k in self.seen for k in
                                        ('lap_count', 'collision_count', 'last_lap_time')},
                'telemetry_fresh': {k: self.fresh(k) for k in ('lap_count', 'collision_count')},
                'counter_baselines': self.baseline}


class RaceProcess:
    def __init__(self, command, env, log):
        self.command, self.env, self.log = command, env, log
        self.process = None

    def start(self):
        self.process = subprocess.Popen(self.command, env=self.env, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True)

    def exited(self):
        return self.process is not None and self.process.poll() is not None

    def stop(self):
        if self.process is None:
            return []
        escalations = []
        # Check the group even if the launch parent has already exited.
        for sig, grace in ((signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                break
            if sig != signal.SIGINT:
                escalations.append(sig.name)
            end = time.monotonic() + grace
            while time.monotonic() < end:
                self.process.poll()  # reap the launch parent
                try:
                    os.killpg(self.process.pid, 0)
                except ProcessLookupError:
                    return escalations
                time.sleep(0.1)
        self.process.wait(timeout=2)
        return escalations


class RosObserver:
    def __init__(self, observations):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, LaserScan
        from std_msgs.msg import Bool, Int32, Float32
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        self.ros, self.obs = rclpy, observations
        # Keep Python's interruption path in charge of cleanup and reporting.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        from rclpy.parameter import Parameter
        # Pose ages against the clock the stack stamps with: the bridge's /clock unless USE_SIM_TIME=false
        # (as common.sh), so a later host clock correction cannot push every pose out of the age window.
        self.node = rclpy.create_node('race_validation_evaluator',
                                      parameter_overrides=[Parameter('use_sim_time', value=sim_time())])
        self.reset_pub = self.node.create_publisher(Bool, '/autodrive/reset_command', 10)
        self.Bool = Bool
        for topic, kind in (('/scan', LaserScan), ('/odom', Odometry), ('/sensors/imu/raw', Imu)):
            self.node.create_subscription(kind, topic, lambda m, t=topic: self.sensor(t), qos_profile_sensor_data)
        self.node.create_subscription(PoseStamped, '/localized_pose', self.pose, 10)
        for name, kind in (('lap_count', Int32), ('collision_count', Int32),
                           ('lap_time', Float32), ('last_lap_time', Float32), ('best_lap_time', Float32)):
            self.node.create_subscription(kind, NS + name, lambda m, k=name: self.metric(k, m.data),
                                          qos_profile_sensor_data)

    def sensor(self, key):
        self.obs.seen[key] = time.monotonic()

    def metric(self, key, value):
        if math.isfinite(value):
            self.obs.record(key, value)

    def pose(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = self.node.get_clock().now().nanoseconds / 1e9 - stamp
        norm = sum(v * v for v in values[3:]) ** 0.5
        if (msg.header.frame_id == 'map' and all(map(math.isfinite, values))
                and abs(norm - 1) <= 0.1 and stamp > 0 and -0.1 <= age <= 0.5):
            self.obs.pose_samples += 1
            self.obs.record('localized_pose', {'stamp_s': stamp, 'frame': 'map', 'xyz_xyzw': values})

    def spin(self):
        self.ros.spin_once(self.node, timeout_sec=0.1)

    def graph(self):
        nodes = [name for name, ns in self.node.get_node_names_and_namespaces()]
        pubs = {t: self.node.count_publishers(t) for t in
                SENSORS | DISABLED | {'/drive', '/localized_pose', '/tracked_pose'}}
        return nodes, pubs

    def reset(self):
        if self.node.count_subscribers('/autodrive/reset_command') != 1:
            raise Failure('expected exactly one simulator reset subscriber')
        for _ in range(3):
            self.reset_pub.publish(self.Bool(data=True))
            self.spin()

    def close(self):
        self.node.destroy_node()
        self.ros.try_shutdown()


def preflight(nodes, pubs, obs):
    conflicts = BLOCKED.intersection(nodes)
    if conflicts or any(pubs.get(t, 0) for t in ('/drive', '/localized_pose', '/tracked_pose')):
        raise Failure('competing controller/localizer: ' + ', '.join(sorted(conflicts)))
    if any(pubs.get(t, 0) for t in DISABLED):
        raise Failure('disabled validation inputs unexpectedly have publishers')
    return (all(nodes.count(n) == 1 for n in BASE_NODES)
            and all(pubs.get(t) == 1 and obs.fresh(t) for t in SENSORS))


def race_health(nodes, pubs, obs, expected, now, started, ready, startup_timeout, exited):
    if exited:
        raise Failure('race launch exited unexpectedly')
    if obs.counter_error:
        raise Failure(obs.counter_error)
    if any(pubs.get(t, 0) for t in DISABLED):
        raise Failure('privileged hint remap has a publisher')
    if any(n in nodes for n in BLOCKED - expected):
        raise Failure('unexpected controller/localizer appeared')
    if (any(nodes.count(n) > 1 for n in BASE_NODES | expected)
            or any(pubs.get(t, 0) > 1 for t in ('/drive', '/localized_pose'))):
        raise Failure('duplicate nodes or competing drive publishers')
    available = (all(nodes.count(n) == 1 for n in BASE_NODES | expected)
                 and pubs.get('/drive') == 1 and pubs.get('/localized_pose') == 1
                 and all(obs.fresh(t) for t in SENSORS) and obs.fresh('localized_pose', 0.5))
    if ready and not available:
        raise Failure('required node/topic lost, sensors stale, or localization missing/invalid')
    if not available and now - started >= startup_timeout:
        raise Failure('startup timeout: missing nodes/topics or valid localization')
    return available


def write_report(directory, report):
    (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    metrics = report['metrics']
    summary = (f"{report['status']}: {report['reason']}\n"
               f"Track: {report['configuration']['track']}; localizer: {report['configuration']['localizer']}; "
               f"speed scale: {report['configuration']['speed_scale']}\n"
               f"Measured laps: {metrics['completed_laps']}; collisions: {metrics['collisions']}; "
               f"localized pose samples: {metrics['pose_samples']}\n"
               'None means unavailable, not zero. Trajectory is a localization estimate.\n')
    (directory / 'summary.txt').write_text(summary)
    print(summary + f'Results: {directory}')


def run(args, observer_factory=RosObserver, process_factory=RaceProcess):
    directory = Path(tempfile.mkdtemp(prefix='race-', dir=prepare_results(args)))
    start = time.monotonic()
    report = {'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'status': 'failed', 'reason': 'setup did not complete', 'artifacts_sha256': {},
              'limitations': ['Native AutoDRIVE must be started manually.',
                             'Existing bridge may use privileged telemetry for crash-triggered safety stops.',
                             'Localizer privileged pose/reset inputs are remapped to unused topics.',
                             'No ground-truth trajectory is consumed; recorded trajectory is estimated.']}
    observer = process = lock = None
    with (directory / 'observations.jsonl').open('w') as sink, (directory / 'race.log').open('w') as log:
        obs = Observations(sink)
        try:
            lock = open(Path(tempfile.gettempdir()) / ('roboracer-validation-' + os.environ.get('ROS_DOMAIN_ID', '0') + '.lock'), 'a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            track = args.tracks_dir / args.track
            files = ['raceline.csv', 'map.yaml'] + (['map.pbstream'] if args.localizer == 'cartographer' else [])
            for name in files:
                report['artifacts_sha256'][name] = hashlib.sha256((track / name).read_bytes()).hexdigest()
            for image in track.glob('*.pgm'):
                report['artifacts_sha256'][image.name] = hashlib.sha256(image.read_bytes()).hexdigest()
            for image in track.glob('*.png'):
                if image.name != 'preview.png':
                    report['artifacts_sha256'][image.name] = hashlib.sha256(image.read_bytes()).hexdigest()
            observer = observer_factory(obs)
            deadline = start + args.timeout
            setup_deadline = min(deadline, start + args.startup_timeout)
            while True:
                observer.spin()
                nodes, pubs = observer.graph()
                if preflight(nodes, pubs, obs):
                    break
                if time.monotonic() >= setup_deadline:
                    raise Failure('preflight timeout: bridge nodes or live sensor topics unavailable')
            report['preflight_nodes'] = nodes
            observer.reset()
            # Let reset frames/counters settle while the vehicle has no controller.
            settle = time.monotonic() + 2
            while time.monotonic() < settle:
                observer.spin()
            nodes, pubs = observer.graph()
            if not preflight(nodes, pubs, obs):
                raise Failure('preflight lost after reset')
            if time.monotonic() >= deadline:
                raise Failure('total timeout during reset/setup')
            obs.baseline = {k: obs.values[k] for k in ('lap_count', 'collision_count') if obs.fresh(k)}
            obs.counter_error = None
            command = ['ros2', 'launch', str(Path(__file__).with_name('validation.launch.py')),
                       'track:=' + args.track, 'localizer:=' + args.localizer,
                       'speed_scale:=' + str(args.speed_scale), 'use_sim_time:=' + str(sim_time()).lower()]
            report['command'] = command
            process = process_factory(command, dict(os.environ, TRACKS_DIR=str(args.tracks_dir)), log)
            process.start()
            started, ready = time.monotonic(), False
            expected = {'pure_pursuit_node'} | ({'mcl'} if args.localizer == 'mcl' else
                        {'cartographer_node', 'cartographer_occupancy_grid_node', 'localization_gate'})
            while True:
                observer.spin()
                now = time.monotonic()
                # DDS graph leases can outlive a child; launch's death messages catch that sooner.
                log.flush()
                if re.search(r'process (?:has died|has finished cleanly)', (directory / 'race.log').read_text()):
                    raise Failure('race child terminated unexpectedly; see race.log')
                nodes, pubs = observer.graph()
                ready = race_health(nodes, pubs, obs, expected, now, started, ready,
                                    args.startup_timeout, process.exited()) or ready
                if obs.delta('collision_count') is not None and obs.delta('collision_count') > 0:
                    raise Failure('collision recorded')
                laps = obs.delta('lap_count')
                if now >= deadline:
                    report.update(status='failed' if laps is not None else 'inconclusive',
                                  reason='total timeout' if laps is not None else
                                  'total timeout; native lap telemetry unavailable, completion unknown')
                    break
                if ready and laps is not None and laps >= args.laps and obs.fresh('lap_count'):
                    verified = obs.fresh('collision_count') and obs.delta('collision_count') == 0
                    report.update(status='passed' if verified else 'inconclusive',
                                  reason='requested native laps completed' if verified else
                                  'laps completed but collision telemetry unavailable/stale')
                    break
        except KeyboardInterrupt:
            report.update(status='interrupted', reason='SIGINT/SIGTERM or keyboard interruption')
        except Exception as error:
            report.update(status='failed', reason=f'{type(error).__name__}: {error}')
        finally:
            # A second signal must not interrupt stop bursts, group cleanup, or report writing.
            handlers = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                report['cleanup_escalations'] = process.stop() if process else []
                if report['cleanup_escalations']:
                    report.update(status='failed', reason='race cleanup required signal escalation')
            except Exception as error:
                report.update(status='failed', reason=f'cleanup failed: {error}')
            finally:
                if observer:
                    try:
                        observer.close()
                    except Exception as error:
                        report.update(status='failed', reason=f'observer cleanup failed: {error}')
                if lock:
                    lock.close()
                report['elapsed_s'] = time.monotonic() - start
                report['metrics'] = obs.metrics()
                try:
                    write_report(directory, report)
                finally:
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)
    return 0 if report['status'] == 'passed' else 130 if report['status'] == 'interrupted' else 1


def prepare_results(args):
    args.results_dir.mkdir(parents=True, exist_ok=True)
    return args.results_dir


def main():
    args = arguments()
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())

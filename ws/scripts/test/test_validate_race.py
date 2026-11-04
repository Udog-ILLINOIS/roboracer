"""Standard-library tests: no ROS, Docker, simulator, or third-party dependencies."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('validate_race', Path(__file__).parents[1] / 'validate_race.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class RunnerTests(unittest.TestCase):
    def test_defaults(self):
        args = v.arguments([])
        self.assertEqual((args.track, args.localizer, args.speed_scale, args.laps),
                         ('test_track_v2', 'mcl', 0.3, 2))

    def test_bad_arguments(self):
        for args in (['--track', '../escape'], ['--laps', '0'], ['--localizer', 'fake'],
                     ['--speed-scale', 'nan'], ['--speed-scale', 'inf'], ['--speed-scale', '1.1'],
                     ['--timeout', '-1'], ['--startup-timeout', '0'],
                     ['--tracks-dir', '/tmp/tracks', '--results-dir', '/tmp/tracks/output']):
            with self.subTest(args=args), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                v.arguments(args)

    def test_configurable(self):
        args = v.arguments(['--track', 'another', '--localizer', 'cartographer', '--laps', '3',
                            '--speed-scale', '.2', '--timeout', '90'])
        self.assertEqual((args.track, args.localizer, args.laps, args.timeout),
                         ('another', 'cartographer', 3, 90))

    def test_missing_metrics_are_null(self):
        obs = v.Observations(io.StringIO(), lambda: 10)
        self.assertIsNone(obs.metrics()['collisions'])
        self.assertIsNone(obs.metrics()['completed_laps'])

    def test_counter_deltas_and_reset_detection(self):
        sink = io.StringIO()
        obs = v.Observations(sink, lambda: 10)
        obs.record('lap_count', 4)
        obs.baseline = {'lap_count': 4}
        obs.record('lap_count', 6)
        self.assertEqual(obs.delta('lap_count'), 2)
        obs.record('lap_count', 0)
        self.assertIn('decreased', obs.counter_error)
        self.assertEqual(len(sink.getvalue().splitlines()), 3)

    def test_preflight_requires_live_sensors_and_no_controllers(self):
        obs = v.Observations(io.StringIO(), lambda: 10)
        pubs = {t: 1 for t in v.SENSORS}
        nodes = list(v.BASE_NODES)
        self.assertFalse(v.preflight(nodes, pubs, obs))
        obs.seen.update({t: 10 for t in v.SENSORS})
        self.assertTrue(v.preflight(nodes, pubs, obs))
        for extra, output in ((['teleop_keyboard'], {}), ([], {'/drive': 1}),
                              ([], {'/validation/disabled_reset_hint': 1})):
            with self.subTest(extra=extra, output=output), self.assertRaises(v.Failure):
                v.preflight(nodes + extra, dict(pubs, **output), obs)

    def test_health_missing_localization_and_child_exit(self):
        obs = v.Observations(io.StringIO(), lambda: 10)
        with self.assertRaisesRegex(v.Failure, 'startup timeout'):
            v.race_health([], {}, obs, {'mcl'}, 10, 0, False, 5, False)
        with self.assertRaisesRegex(v.Failure, 'launch exited'):
            v.race_health([], {}, obs, {'mcl'}, 10, 0, False, 5, True)
        with self.assertRaisesRegex(v.Failure, 'lost'):
            v.race_health([], {}, obs, {'mcl'}, 10, 0, True, 5, False)

    def test_running_health_checks_freshness_and_competition(self):
        obs = v.Observations(io.StringIO(), lambda: 10)
        obs.seen.update({t: 10 for t in v.SENSORS | {'localized_pose'}})
        expected = {'mcl', 'pure_pursuit_node'}
        nodes = list(v.BASE_NODES | expected)
        pubs = {'/drive': 1, '/localized_pose': 1}
        self.assertTrue(v.race_health(nodes, pubs, obs, expected, 10, 0, True, 5, False))
        obs.seen['/scan'] = 0
        with self.assertRaisesRegex(v.Failure, 'stale'):
            v.race_health(nodes, pubs, obs, expected, 10, 0, True, 5, False)
        obs.seen['/scan'] = 10
        for changed_nodes, changed_pubs in (
                (nodes + ['teleop_keyboard'], pubs),
                (nodes + ['mcl'], pubs),
                (nodes, dict(pubs, **{'/drive': 2})),
                (nodes, dict(pubs, **{'/validation/disabled_ground_truth': 1}))):
            with self.subTest(nodes=changed_nodes, pubs=changed_pubs), self.assertRaises(v.Failure):
                v.race_health(changed_nodes, changed_pubs, obs, expected, 10, 0, True, 5, False)

    def scenario(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            track = root / 'tracks/test_track_v2'
            track.mkdir(parents=True)
            for name in ('raceline.csv', 'map.yaml'):
                (track / name).write_text('fixture, never modified')
            args = v.arguments(['--tracks-dir', str(root / 'tracks'), '--results-dir', str(root / 'results'),
                                '--timeout', '15', '--startup-timeout', '3'])
            state = {'started': False, 'stopped': False, 'closed': False, 'tick': 0, 'reset': False}
            class Clock:
                now = 0
                def __call__(self):
                    self.now += .01
                    return self.now
            clock = Clock()
            class Observer:
                def __init__(self, obs):
                    self.obs = obs
                def spin(self):
                    for t in v.SENSORS:
                        self.obs.seen[t] = clock()
                    if mode != 'no_laps':
                        self.obs.record('lap_count', 0 if not state['started'] else
                                        (0 if mode == 'timeout' else 2))
                    if mode != 'no_collision_telemetry':
                        self.obs.record('collision_count', int(state['started'] and mode == 'collision'))
                    if state['started'] and mode != 'no_localization':
                        self.obs.record('localized_pose', {'frame': 'map'})
                        self.obs.pose_samples += 1
                def graph(self):
                    nodes = list(v.BASE_NODES)
                    pubs = {t: 1 for t in v.SENSORS}
                    if state['started']:
                        nodes += ['mcl', 'pure_pursuit_node']
                        pubs.update({'/drive': 1, '/localized_pose': 1})
                    return nodes, pubs
                def reset(self):
                    state['reset'] = True
                def close(self):
                    state['closed'] = True
            class Process:
                def __init__(self, command, env, log):
                    self.log = log
                def start(self):
                    state['started'] = True
                    if mode == 'interrupt':
                        raise KeyboardInterrupt
                    if mode == 'child_death':
                        self.log.write('process has died [pid 123, exit code -9]\n')
                def exited(self):
                    return mode == 'launch_exit'
                def stop(self):
                    state['stopped'] = True
                    return []
            # Observations' default clock was bound at import; replace it explicitly.
            original = v.Observations
            with patch.object(v.time, 'monotonic', clock), patch.object(v, 'Observations',
                    lambda sink: original(sink, clock)), patch('sys.stdout', io.StringIO()):
                code = v.run(args, Observer, Process)
            report_dir = next((root / 'results').iterdir())
            report = json.loads((report_dir / 'report.json').read_text())
            self.assertTrue((report_dir / 'summary.txt').is_file())
            self.assertTrue(state['closed'] and state['stopped'] and state['reset'])
            self.assertEqual((track / 'raceline.csv').read_text(), 'fixture, never modified')
            return code, report

    def test_lifecycle_and_report_success(self):
        code, report = self.scenario('success')
        self.assertEqual(code, 0)
        self.assertEqual(report['metrics']['completed_laps'], 2)
        self.assertEqual(report['metrics']['collisions'], 0)
        self.assertEqual(report['status'], 'passed')
        self.assertIn('raceline.csv', report['artifacts_sha256'])

    def test_failures_cleanup_and_reporting(self):
        for mode, reason in [('collision', 'collision'), ('no_localization', 'startup timeout'),
                             ('timeout', 'total timeout'), ('launch_exit', 'launch exited'),
                             ('child_death', 'child terminated'), ('interrupt', 'interruption')]:
            with self.subTest(mode=mode):
                code, report = self.scenario(mode)
                self.assertNotEqual(code, 0)
                self.assertIn(reason, report['reason'])

    def test_missing_collision_data_cannot_pass(self):
        code, report = self.scenario('no_collision_telemetry')
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertIsNone(report['metrics']['collisions'])

    def test_missing_lap_telemetry_times_out_inconclusive(self):
        code, report = self.scenario('no_laps')
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertIsNone(report['metrics']['completed_laps'])
        self.assertIn('completion unknown', report['reason'])

    def test_missing_track_still_reports(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sys.stdout', io.StringIO()):
            args = v.arguments(['--tracks-dir', tmp + '/missing', '--results-dir', tmp + '/results'])
            self.assertEqual(v.run(args), 1)
            report = json.loads(next((Path(tmp) / 'results').glob('*/report.json')).read_text())
            self.assertIn('FileNotFoundError', report['reason'])

    def test_real_process_group_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'ready'
            code = ('import signal,time,pathlib; '
                    'signal.signal(signal.SIGINT, lambda *_: exit(0)); '
                    f'pathlib.Path({str(marker)!r}).touch(); time.sleep(60)')
            with open(os.devnull, 'w') as log:
                process = v.RaceProcess([sys.executable, '-c', code], dict(os.environ), log)
                process.start()
                try:
                    end = time.monotonic() + 5
                    while not marker.exists() and time.monotonic() < end:
                        time.sleep(.01)
                    self.assertTrue(marker.exists())
                    self.assertEqual(process.stop(), [])
                    self.assertTrue(process.exited())
                finally:
                    if not process.exited():
                        os.killpg(process.process.pid, signal.SIGKILL)
                        process.process.wait()


if __name__ == '__main__':
    unittest.main()

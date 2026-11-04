"""Synthetic report tests; no ROS, Docker, simulator, or third-party dependencies."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('analyze_validation', Path(__file__).parents[1] / 'analyze_validation.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def report(status='passed'):
    return {'schema_version': 1, 'started_utc': '2026-11-01T12:00:00+00:00',
            'elapsed_s': 50, 'status': status, 'reason': 'requested native laps completed',
            'configuration': {'track': 'test_track_v2', 'localizer': 'mcl', 'speed_scale': .3},
            'artifacts_sha256': {'raceline.csv': 'identical', 'map.yaml': 'identical'},
            'metrics': {'completed_laps': 3, 'collisions': 0, 'pose_samples': 100,
                        'counter_baselines': {'lap_count': 0},
                        'telemetry_available': {'lap_count': True, 'collision_count': True, 'last_lap_time': True},
                        'telemetry_fresh': {'lap_count': True, 'collision_count': True}}}


def row(t, topic, value):
    return {'monotonic_s': t, 'topic': topic, 'value': value}


def timed_records(times=(20, 22)):
    records = [row(0, 'lap_count', 7), row(.1, 'last_lap_time', 99),
               row(.2, 'lap_count', 0),  # pre-race reset, not a bad racing counter
               row(1, 'localized_pose', {}), row(1.1, 'lap_count', 0),
               row(3, 'lap_count', 1), row(3.1, 'last_lap_time', 4), row(3.2, 'last_lap_time', 4)]
    for count, value in enumerate(times, 2):
        stamp = count * 10
        records.extend([row(stamp, 'lap_count', count), row(stamp + .1, 'last_lap_time', value),
                        row(stamp + .2, 'last_lap_time', value)])
    return records


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def fixture(self, name='race-one', data=None, records=None):
        directory = self.root / name
        directory.mkdir()
        (directory / 'report.json').write_text(json.dumps(data if data is not None else report()))
        if records is not None:
            (directory / 'observations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        return directory

    def test_latest_means_finalized_not_successful_or_last_modified(self):
        older = self.fixture('older')
        newer_data = report('failed')
        newer_data['elapsed_s'] = 90
        newer = self.fixture('newer', newer_data)
        os.utime(older / 'report.json', (9999999999, 9999999999))
        (self.root / 'unfinished').mkdir()
        broken = self.root / 'broken'
        broken.mkdir()
        (broken / 'report.json').write_text('{')
        selected, warnings = a.latest(self.root)
        self.assertEqual(selected, newer / 'report.json')
        self.assertTrue(any('Skipped' in warning for warning in warnings))

    def test_latest_mtime_fallback(self):
        data = report()
        del data['started_utc']
        directory = self.fixture(data=data)
        selected, warnings = a.latest(self.root)
        self.assertEqual(selected, directory / 'report.json')
        self.assertIn('mtime', warnings[0])

    def test_no_completed_reports(self):
        with self.assertRaisesRegex(ValueError, 'No completed'):
            a.latest(self.root)

    def test_summary_fields_and_missing_observations(self):
        run = a.analyze(self.fixture())
        text = a.summary(run)
        for expected in ('PASSED', 'Completed laps: 3', 'collisions: 0', 'localization samples: 100',
                         'localizer: mcl', 'speed scale: 0.3', 'requested native laps completed',
                         'Observations unavailable'):
            self.assertIn(expected, text)

    def test_alerts_and_nulls(self):
        data = report('failed')
        data['reason'] = 'localization missing/invalid'
        data['metrics'].update(collisions=2, pose_samples=0, completed_laps=None,
                               telemetry_available={}, telemetry_fresh={})
        run = a.analyze(self.fixture(data=data, records=[]))
        text = a.summary(run)
        for expected in ('RUN FAILED', 'COLLISIONS: 2', 'LOCALIZATION', 'Missing/unknown telemetry',
                         'Completed laps: unavailable'):
            self.assertIn(expected, text)
        self.assertIsNone(run['timing']['statistics'])

    def test_lap_stats_exclude_setup_first_crossing_and_broadcast_duplicates(self):
        run = a.analyze(self.fixture(records=timed_records()))
        timing = run['timing']
        self.assertEqual(timing['full_lap_candidates_s'], [20, 22])
        self.assertEqual(timing['statistics']['count'], 2)
        self.assertEqual(timing['statistics']['mean_s'], 21)
        self.assertEqual(timing['statistics']['median_s'], 21)
        self.assertEqual(timing['statistics']['min_s'], 20)
        self.assertEqual(timing['statistics']['max_s'], 22)
        self.assertEqual(timing['native']['last_lap_time']['positive_observations'], 7)

    def test_identical_lap_times_remain_two_distinct_laps(self):
        timing = a.lap_analysis(timed_records((20, 20)), report())
        self.assertEqual(timing['statistics']['count'], 2)
        self.assertEqual(timing['statistics']['sample_stddev_s'], 0)

    def test_insufficient_time_data_has_no_statistics(self):
        for records in ([], timed_records((20,)), [row(0, 'last_lap_time', 12)]):
            with self.subTest(records=records):
                self.assertIsNone(a.lap_analysis(records, report())['statistics'])

    def test_counter_jumps_resets_and_ambiguous_associations_suppress_stats(self):
        records = timed_records()
        records.append(row(40, 'lap_count', 0))
        self.assertIsNone(a.lap_analysis(records, report())['statistics'])
        records = timed_records()
        records[-1]['value'] = 23  # last_lap_time changed within the association window
        self.assertIsNone(a.lap_analysis(records, report())['statistics'])
        records = timed_records()
        records[4]['value'] = 1  # missed baseline
        self.assertIsNone(a.lap_analysis(records, report())['statistics'])

    def test_malformed_truncated_or_nonmonotonic_records_warn(self):
        directory = self.fixture(records=timed_records())
        with (directory / 'observations.jsonl').open('a') as stream:
            stream.write('{"unfinished":\n')
            stream.write(json.dumps(row(-1, 'lap_count', 0)) + '\n')
        run = a.analyze(directory)
        self.assertIsNone(run['timing']['statistics'])
        self.assertTrue(any('observations line' in x for x in run['alerts']))

    def test_comparison_available_metrics_and_configuration(self):
        left = a.analyze(self.fixture('left', records=timed_records()))
        data = report()
        data['configuration']['speed_scale'] = .5
        data['metrics'].update(completed_laps=4, collisions=None)
        right = a.analyze(self.fixture('right', data, timed_records((18, 20))))
        rows, warnings = a.comparison(left, right)
        lookup = {r[0]: r for r in rows}
        self.assertEqual(lookup['completed_laps'][3], 1)
        self.assertIsNone(lookup['collisions'][3])
        self.assertEqual(lookup['lap_statistics.mean_s'][3], -2)
        self.assertTrue(any('speed_scale' in w for w in warnings))
        right['report']['artifacts_sha256'] = {}
        rows, warnings = a.comparison(left, right)
        self.assertIsNone(dict((r[0], r[3]) for r in rows)['lap_statistics.mean_s'])

    def test_csv_no_overwrite_and_preserve_inputs(self):
        directory = self.fixture(records=timed_records())
        before = {p: p.read_bytes() for p in directory.iterdir()}
        run = a.analyze(directory)
        rows, _ = a.comparison(run, run)
        output = self.root / 'comparison.csv'
        a.export_csv(output, rows, [run])
        self.assertIn('delta_b_minus_a', output.read_text())
        with self.assertRaises(FileExistsError):
            a.export_csv(output, rows, [run])
        with self.assertRaises(ValueError):
            a.export_csv(directory / 'new.csv', rows, [run])
        run['report']['configuration']['tracks_dir'] = str(self.root / 'tracks')
        with self.assertRaises(ValueError):
            a.export_csv(self.root / 'tracks/new.csv', rows, [run])
        self.assertEqual(before, {p: p.read_bytes() for p in directory.iterdir()})

    def test_cli_latest_compare_and_csv(self):
        first = self.fixture('first', records=timed_records())
        second = self.fixture('second', records=timed_records())
        with patch('sys.stdout', io.StringIO()) as output:
            self.assertEqual(a.main(['--results-dir', str(self.root)]), 0)
            self.assertIn('Outcome: PASSED', output.getvalue())
        with patch('sys.stdout', io.StringIO()) as output:
            self.assertEqual(a.main(['--results-dir', str(self.root), '--compare', first.name, second.name,
                                     '--csv', str(self.root / 'out.csv')]), 0)
            self.assertIn('Comparison (B - A', output.getvalue())
        with patch('sys.stderr', io.StringIO()):
            self.assertEqual(a.main(['--run', str(self.root / 'missing')]), 2)
            with self.assertRaises(SystemExit):
                a.main(['--csv', str(self.root / 'invalid.csv')])

    def test_reject_partial_or_unknown_schema(self):
        for field, value in (('status', 'running'), ('schema_version', 2), ('metrics', None), ('elapsed_s', None)):
            data = report()
            data[field] = value
            path = self.root / 'report.json'
            path.write_text(json.dumps(data))
            with self.subTest(field=field), self.assertRaises(ValueError):
                a.load_report(path)


if __name__ == '__main__':
    unittest.main()

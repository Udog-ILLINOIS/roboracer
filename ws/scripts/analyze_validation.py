#!/usr/bin/env python3
"""Read-only validation analysis; no ROS or simulator dependencies."""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
import sys

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'log/validation'
TERMINAL = {'passed', 'failed', 'inconclusive', 'interrupted'}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_report(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'report.json'
    with path.open() as stream:
        report = json.load(stream)
    if (not isinstance(report, dict) or report.get('schema_version') != 1
            or not isinstance(report.get('status'), str) or report.get('status') not in TERMINAL
            or not isinstance(report.get('metrics'), dict)
            or not isinstance(report.get('configuration'), dict)
            or not number(report.get('elapsed_s')) or report['elapsed_s'] < 0):
        raise ValueError(f'{path}: not a finalized supported validation report')
    return path.resolve(), report


def latest(root):
    """Final reports include failures/interruptions; directories without one are unfinished."""
    candidates, warnings = [], []
    for path in sorted(Path(root).glob('*/report.json')):
        try:
            path, report = load_report(path)
            # Runner has no finished_utc: estimate completion from start + elapsed.
            try:
                start = datetime.fromisoformat(report['started_utc'])
                if start.tzinfo is None:
                    raise ValueError('timestamp lacks timezone')
                end = start.timestamp() + report['elapsed_s']
            except (KeyError, TypeError, ValueError, OverflowError):
                end = path.stat().st_mtime
                warnings.append(f'{path.parent.name}: using report mtime; completion timestamp unavailable')
            candidates.append((end, str(path), path))
        except (OSError, ValueError) as error:
            warnings.append(f'Skipped {path}: {error}')
    if not candidates:
        raise ValueError(f'No completed validation reports under {root}')
    return max(candidates)[2], warnings


def observations(path):
    records, warnings = [], []
    try:
        with path.open() as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    if (not isinstance(record, dict) or not number(record.get('monotonic_s'))
                            or not isinstance(record.get('topic'), str) or 'value' not in record):
                        raise ValueError('invalid observation fields')
                    if records and record['monotonic_s'] < records[-1]['monotonic_s']:
                        raise ValueError('nonmonotonic observation time')
                    records.append(record)
                except (TypeError, ValueError) as error:
                    warnings.append(f'observations line {line_no}: {error}')
    except OSError as error:
        warnings.append(f'Observations unavailable: {error}')
    return records, warnings


def lap_analysis(records, report, damaged=False):
    """Conservative receive-time association; repeated broadcasts are not separate laps.

    Require consecutive counter transitions after the first localized pose, an
    observed baseline, and stable last_lap_time samples 0.05..0.5 s after crossing.
    Skip the first crossing (possibly partial). Never infer times from trajectory.
    """
    native = {}
    for key in ('lap_time', 'last_lap_time', 'best_lap_time'):
        values = [r['value'] for r in records if r['topic'] == key and number(r['value']) and r['value'] > 0]
        native[key] = {'positive_observations': len(values), 'last_positive_s': values[-1] if values else None}
        if key != 'lap_time':
            native[key]['positive_value_changes_s'] = [value for i, value in enumerate(values)
                                                      if i == 0 or value != values[i - 1]]
    notes, samples, transitions = [], [], []
    poses = [r['monotonic_s'] for r in records if r['topic'] == 'localized_pose']
    baselines = report['metrics'].get('counter_baselines', {})
    baseline = baselines.get('lap_count') if isinstance(baselines, dict) else None
    if not poses or not number(baseline) or baseline < 0 or baseline != int(baseline):
        notes.append('No localized observation/baseline for lap-time association')
    else:
        counter = None
        for r in records:
            if r['monotonic_s'] < poses[0] or r['topic'] != 'lap_count':
                continue
            value = r['value']
            if not number(value) or value < 0 or value != int(value):
                damaged = True
                continue
            if counter is None:
                if value != baseline:
                    notes.append('Post-localization lap baseline not observed')
                    damaged = True
            elif value != counter:
                if value != counter + 1:
                    notes.append('Lap counter reset/jump: timing statistics suppressed')
                    damaged = True
                transitions.append((r['monotonic_s'], value))
            counter = value
        for index, (stamp, count) in enumerate(transitions):
            if count <= baseline + 1:  # first crossing may be a partial lap
                continue
            end = min(stamp + .5, transitions[index + 1][0] if index + 1 < len(transitions) else stamp + .5)
            values = [r['value'] for r in records if r['topic'] == 'last_lap_time'
                      and stamp + .05 <= r['monotonic_s'] < end and number(r['value']) and r['value'] > 0]
            if len(values) >= 2 and max(values) - min(values) <= 1e-5:
                samples.append(values[-1])
            else:
                notes.append(f'Lap counter {count}: missing or ambiguous last_lap_time association')
    stats = None
    if damaged:
        notes.append('Damaged/discontinuous observations: timing statistics suppressed')
    elif len(samples) >= 2:
        stats = {'count': len(samples), 'mean_s': statistics.mean(samples),
                 'median_s': statistics.median(samples), 'min_s': min(samples), 'max_s': max(samples),
                 'sample_stddev_s': statistics.stdev(samples)}
    else:
        notes.append('Insufficient full-lap candidates: need at least two for statistics')
    return {'native': native, 'full_lap_candidates_s': samples, 'statistics': stats, 'notes': notes}


def analyze(path):
    path, report = load_report(path)
    records, warnings = observations(path.with_name('observations.jsonl'))
    timing = lap_analysis(records, report, damaged=bool(warnings))
    metrics = report['metrics']
    alerts = list(warnings)
    if report['status'] != 'passed':
        alerts.append('RUN ' + report['status'].upper() + ': ' + str(report.get('reason', 'unknown reason')))
    for name in ('completed_laps', 'collisions', 'pose_samples'):
        if not number(metrics.get(name)) or metrics[name] < 0:
            alerts.append(f'Missing/invalid metric: {name}')
    if number(metrics.get('collisions')) and metrics['collisions'] > 0:
        alerts.append(f"COLLISIONS: {metrics['collisions']}")
    reason = str(report.get('reason', ''))
    if metrics.get('pose_samples') == 0:
        alerts.append('LOCALIZATION: no valid localized samples recorded')
    elif report['status'] != 'passed' and any(word in reason.lower() for word in ('localiz', 'pose')):
        alerts.append('LOCALIZATION issue mentioned by runner; reason may also include other missing inputs')
    available, fresh = metrics.get('telemetry_available', {}), metrics.get('telemetry_fresh', {})
    for key in ('lap_count', 'collision_count', 'last_lap_time'):
        if not isinstance(available, dict) or available.get(key) is not True:
            alerts.append(f'Missing/unknown telemetry: {key}')
        if key != 'last_lap_time' and (not isinstance(fresh, dict) or fresh.get(key) is not True):
            alerts.append(f'Stale/unknown telemetry freshness: {key}')
    alerts.extend(timing['notes'])
    return {'path': path, 'report': report, 'timing': timing, 'alerts': alerts}


def display(value):
    return 'unavailable' if value is None else str(value)


def summary(run):
    r, t = run['report'], run['timing']
    c, m = r['configuration'], r['metrics']
    lines = [f"Run: {run['path'].parent.name} ({run['path']})", f"Outcome: {r['status'].upper()}",
             f"Track: {display(c.get('track'))}; localizer: {display(c.get('localizer'))}; speed scale: {display(c.get('speed_scale'))}",
             f"Completed laps: {display(m.get('completed_laps'))}; collisions: {display(m.get('collisions'))}; localization samples: {display(m.get('pose_samples'))}",
             f"Reason: {display(r.get('reason'))}"]
    for key, value in t['native'].items():
        lines.append(f"Native {key}: last positive={display(value['last_positive_s'])} s; broadcasts={value['positive_observations']} (not lap count)")
        if 'positive_value_changes_s' in value:
            lines.append(f"  Raw value changes (includes setup; not independent laps): {value['positive_value_changes_s']}")
    lines.append(f"Full-lap candidates (receive-time association; first crossing excluded): {t['full_lap_candidates_s']}")
    lines.append('Lap statistics: ' + (str(t['statistics']) if t['statistics'] else 'unavailable'))
    lines.extend('! ' + alert for alert in run['alerts'])
    lines.extend('Limitation: ' + str(item) for item in r.get('limitations', []))
    return '\n'.join(lines)


def comparison(a, b):
    rows, warnings = [], []
    for key in ('track', 'localizer', 'speed_scale'):
        left, right = a['report']['configuration'].get(key), b['report']['configuration'].get(key)
        rows.append((key, left, right, None))
        if left != right:
            warnings.append(f'Configuration differs: {key} ({display(left)} -> {display(right)})')
    for key in ('status', 'reason'):
        rows.append((key, a['report'].get(key), b['report'].get(key), None))
    for key in ('completed_laps', 'collisions', 'pose_samples'):
        left, right = a['report']['metrics'].get(key), b['report']['metrics'].get(key)
        rows.append((key, left, right, right - left if number(left) and number(right)
                     and left >= 0 and right >= 0 else None))
    for key in ('lap_count', 'collision_count', 'last_lap_time'):
        for field in ('telemetry_available', 'telemetry_fresh'):
            left = a['report']['metrics'].get(field, {})
            right = b['report']['metrics'].get(field, {})
            rows.append((f'{field}.{key}', left.get(key) if isinstance(left, dict) else None,
                         right.get(key) if isinstance(right, dict) else None, None))
    hashes_a, hashes_b = a['report'].get('artifacts_sha256', {}), b['report'].get('artifacts_sha256', {})
    track_a = a['report']['configuration'].get('track')
    compatible = (track_a is not None and track_a == b['report']['configuration'].get('track')
                  and bool(hashes_a) and hashes_a == hashes_b)
    if not compatible:
        warnings.append('Track/artifact identity differs or is unknown; lap-time deltas suppressed')
    for key in ('count', 'mean_s', 'median_s', 'min_s', 'max_s', 'sample_stddev_s'):
        left = (a['timing']['statistics'] or {}).get(key)
        right = (b['timing']['statistics'] or {}).get(key)
        rows.append(('lap_statistics.' + key, left, right,
                     right - left if compatible and number(left) and number(right) else None))
    if not a['timing']['statistics'] or not b['timing']['statistics']:
        warnings.append('Insufficient timing data for a lap-statistics comparison')
    if any(r['report']['status'] != 'passed' for r in (a, b)):
        warnings.append('Comparison includes a non-passing run; differences are descriptive, not a performance verdict')
    return rows, warnings


def export_csv(path, rows, runs):
    path = Path(path).resolve()
    protected = [Path(__file__).resolve().parents[1] / 'tracks',
                 Path(os.environ.get('TRACKS_DIR', '~/ws/tracks')).expanduser().resolve()]
    protected += [r['path'].parent for r in runs]
    protected += [Path(r['report']['configuration']['tracks_dir']).expanduser().resolve()
                  for r in runs if r['report']['configuration'].get('tracks_dir')]
    if any(path == root or root in path.parents for root in protected):
        raise ValueError('CSV must be outside run directories and validated track directories')
    # Exclusive creation also refuses existing outputs and symlinks; no overwrite switch.
    with path.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric', 'run_a', 'run_b', 'delta_b_minus_a'])
        writer.writerows(rows)  # None becomes an empty field, never zero.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_ROOT)
    select = parser.add_mutually_exclusive_group()
    select.add_argument('--run', help='run directory/report path, or name under results-dir')
    select.add_argument('--compare', nargs=2, metavar=('RUN_A', 'RUN_B'))
    parser.add_argument('--csv', type=Path, help='new comparison CSV path (never overwrite)')
    args = parser.parse_args(argv)
    if args.csv and not args.compare:
        parser.error('--csv requires --compare')
    def resolve(value):
        p = Path(value).expanduser()
        return p if p.exists() else args.results_dir / p
    try:
        if args.compare:
            runs = [analyze(resolve(value)) for value in args.compare]
            rows, warnings = comparison(*runs)
            print('\n\n'.join(summary(run) for run in runs))
            print('\nComparison (B - A; unavailable values have no delta):')
            for key, left, right, delta in rows:
                print(f'  {key}: {display(left)} -> {display(right)}; delta={display(delta)}')
            for warning in warnings:
                print('! ' + warning)
            if args.csv:
                export_csv(args.csv, rows, runs)
                print(f'CSV: {args.csv}')
        else:
            if args.run:
                path, warnings = resolve(args.run), []
            else:
                path, warnings = latest(args.results_dir)
            for warning in warnings:
                print('! ' + warning)
            print(summary(analyze(path)))
        return 0  # Analysis success, not a new racing pass/fail decision.
    except (OSError, ValueError, TypeError) as error:
        print(f'Analysis error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

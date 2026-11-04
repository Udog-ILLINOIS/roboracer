"""Read-only occupancy/trajectory/graph diagnostics. Never relax planner checks."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from PIL import Image
import yaml


def load_trajectory(path):
    data = np.loadtxt(path, delimiter=',', comments='#', ndmin=2)
    if data.shape[1] != 4 or not len(data) or not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f'invalid t,x,y,yaw trajectory: {path}')
    return data


def map_data(path):
    meta = yaml.safe_load(Path(path).read_text())
    res, origin = float(meta['resolution']), np.asarray(meta['origin'], dtype=float)
    if not math.isfinite(res) or res <= 0 or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('invalid map resolution/origin')
    image = Path(meta['image'])
    if not image.is_absolute(): image = Path(path).parent / image
    pixels = np.asarray(Image.open(image).convert('L'))
    probability = pixels / 255.0 if meta.get('negate', 0) else (255.0 - pixels) / 255.0
    occupied = probability > meta.get('occupied_thresh', .65)
    free = probability < meta.get('free_thresh', .25)
    if meta.get('mode', 'trinary') == 'trinary' and not meta.get('negate', 0): free &= pixels != 205
    return meta, pixels, free, occupied


def occupancy(traj, meta, free, occupied):
    xy = traj[:, 1:3] - np.asarray(meta['origin'][:2])
    yaw = meta['origin'][2]
    c, s = math.cos(yaw), math.sin(yaw)
    local = xy @ np.array([[c, -s], [s, c]])
    col = np.floor(local[:, 0] / meta['resolution']).astype(int)
    row = free.shape[0] - 1 - np.floor(local[:, 1] / meta['resolution']).astype(int)
    inside = (col >= 0) & (col < free.shape[1]) & (row >= 0) & (row < free.shape[0])
    f, o = np.zeros(len(traj), bool), np.zeros(len(traj), bool)
    f[inside], o[inside] = free[row[inside], col[inside]], occupied[row[inside], col[inside]]
    masks = {'free': f, 'occupied': o, 'unknown': inside & ~f & ~o, 'outside': ~inside}
    return {key: {'count': int(mask.sum()), 'fraction': float(mask.mean())} for key, mask in masks.items()}


def motion(traj):
    dt = np.diff(traj[:, 0])
    ds = np.linalg.norm(np.diff(traj[:, 1:3], axis=0), axis=1)
    turns = np.arctan2(np.sin(np.diff(traj[:, 3])), np.cos(np.diff(traj[:, 3])))
    # Diagnostic flags only, not planner thresholds: gross >0.5 m steps or >3 m/s.
    flagged = np.flatnonzero((ds > .5) | (ds / dt > 3) | (abs(turns) > .5))
    return {'samples': len(traj), 'duration_s': float(traj[-1, 0] - traj[0, 0]),
            'length_m': float(ds.sum()), 'max_step_m': float(ds.max()) if len(ds) else None,
            'max_sample_gap_s': float(dt.max()) if len(dt) else None,
            'total_heading_change_rad': float(turns.sum()),
            'discontinuities': [{'index': int(i), 't': float(traj[i+1, 0]), 'distance_m': float(ds[i]),
                                 'dt_s': float(dt[i]), 'heading_step_rad': float(turns[i])} for i in flagged]}


def match_corrections(online, final):
    indices = np.searchsorted(final[:, 0], online[:, 0])
    indices = np.clip(indices, 0, len(final)-1)
    before = np.maximum(indices-1, 0)
    indices = np.where(abs(final[before, 0]-online[:, 0]) < abs(final[indices, 0]-online[:, 0]), before, indices)
    valid = abs(final[indices, 0] - online[:, 0]) <= 1e-4
    distances = np.linalg.norm(final[indices[valid], 1:3] - online[valid, 1:3], axis=1)
    angles = final[indices[valid], 3] - online[valid, 3]
    return {'timestamp_matches': int(valid.sum()), 'unmatched_online': int((~valid).sum()),
            'median_position_correction_m': float(np.median(distances)) if len(distances) else None,
            'max_position_correction_m': float(distances.max()) if len(distances) else None,
            'corrections_over_0_5_m': int((distances > .5).sum()),
            'max_heading_correction_rad': float(abs(np.arctan2(np.sin(angles), np.cos(angles))).max()) if len(angles) else None}


def log_summary(path):
    if path is None: return {'available': False, 'warnings': None}
    try:
        text = Path(path).read_text(errors='replace')
    except OSError as error:
        return {'available': False, 'warnings': None, 'error': str(error)}
    lines = text.splitlines()
    warnings = [line for line in lines if re.search(r'WARN|ERROR|queue is full|extrapolat|Bounding to maximum', line, re.I)]
    return {'available': True, 'queue_full_drops': text.count('queue is full'),
            'thread_limit_warnings': text.count('Bounding to maximum'), 'warnings': warnings}


def compare_rasters(saved, native):
    sm, sp, _, _ = map_data(saved)
    nm, npixels, _, _ = map_data(native)
    same_geometry = (sp.shape == npixels.shape and math.isclose(sm['resolution'], nm['resolution'], abs_tol=1e-7)
                     and np.allclose(sm['origin'], nm['origin'], atol=1e-6, rtol=0))
    return {'geometry_matches': bool(same_geometry),
            'different_pixels': int((sp != npixels).sum()) if same_geometry else None,
            'different_fraction': float((sp != npixels).mean()) if same_geometry else None}


def diagnose(map_yaml, online_path, final_path=None, odometry_path=None, mapping_log=None, follower_log=None):
    meta, pixels, free, occupied = map_data(map_yaml)
    online = load_trajectory(online_path)
    result = {'schema_version': 1,
              'map': {'resolution_m': meta['resolution'], 'origin': meta['origin'],
                      'width': pixels.shape[1], 'height': pixels.shape[0],
                      'encoding': {str(int(v)): int(n) for v, n in zip(*np.unique(pixels, return_counts=True))},
                      'standard_trinary_encoding': bool(np.isin(pixels, [0,205,254]).all()),
                      'occupied_thresh': meta.get('occupied_thresh'), 'free_thresh': meta.get('free_thresh')},
              'online': {'occupancy': occupancy(online, meta, free, occupied), 'motion': motion(online)},
              'mapping_log': log_summary(mapping_log), 'follower_log': log_summary(follower_log),
              'interpretation': []}
    final = None
    if final_path:
        final = load_trajectory(final_path)
        result['final'] = {'occupancy': occupancy(final, meta, free, occupied), 'motion': motion(final)}
        result['online_to_final'] = match_corrections(online, final)
        graph_map = Path(final_path).with_name('map.yaml')
        if graph_map.exists() and graph_map.resolve() != Path(map_yaml).resolve():  # not a file with itself
            result['graph_raster_comparison'] = compare_rasters(map_yaml, graph_map)
        if result['online_to_final']['corrections_over_0_5_m']:
            result['interpretation'].append('Historical online poses received large later graph corrections.')
        bad = sum(result['final']['occupancy'][k]['count'] for k in ('occupied','unknown','outside'))
        if bad or result['final']['motion']['discontinuities']:
            result['interpretation'].append('Final graph still has occupancy/discontinuity warnings; not merely stale history.')
        else:
            result['interpretation'].append('Final samples agree with map cells; this alone does not establish correct topology or loop closure.')
        # Run existing geometry checks unchanged, without writing a raceline.
        try:
            from planning.centerline import extract_centerline
            extract_centerline(str(map_yaml), str(final_path))
            result['planner_geometry'] = {'passed': True, 'reason': None}
        except ImportError:
            result['planner_geometry'] = {'passed': None, 'reason': 'planning package unavailable'}
        except Exception as error:  # a diagnostic: never abort a save that would otherwise succeed
            result['planner_geometry'] = {'passed': False, 'reason': f'{type(error).__name__}: {error}'}
            result['interpretation'].append('Existing planner still rejects final graph geometry.')
    if odometry_path:
        odom = load_trajectory(odometry_path)
        result['odometry'] = motion(odom)
        result['interpretation'].append('Stored odometry is a sensor estimate in its own frame, not independent ground truth or a calibration reference.')
    return result


def concise(report):
    lines = [f"Map: {report['map']['width']}x{report['map']['height']}, {report['map']['resolution_m']} m/pixel, origin={report['map']['origin']}"]
    for key in ('online', 'final'):
        if key not in report: continue
        values = report[key]['occupancy']
        lines.append(key + ': ' + ', '.join(f"{k}={v['count']} ({v['fraction']:.1%})" for k,v in values.items()))
        lines.append(f"  max step={report[key]['motion']['max_step_m']} m; gross discontinuities={len(report[key]['motion']['discontinuities'])}")
    if 'online_to_final' in report: lines.append('Corrections: ' + str(report['online_to_final']))
    if 'planner_geometry' in report: lines.append('Planner geometry: ' + str(report['planner_geometry']))
    if 'graph_raster_comparison' in report: lines.append('Native raster comparison: ' + str(report['graph_raster_comparison']))
    logs = report['mapping_log']
    lines.append(f"Mapping log: available={logs['available']}; queue-full drops={logs.get('queue_full_drops')}; thread-limit warnings={logs.get('thread_limit_warnings')}")
    lines.extend('! ' + item for item in report['interpretation'])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('track_dir', type=Path)
    parser.add_argument('--export-graph', action='store_true', help='read native snapshot using installed SDK; never modifies input')
    parser.add_argument('--mapping-log', type=Path)
    parser.add_argument('--follower-log', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path.home()/'ws/log/slam_diagnostics')
    args = parser.parse_args(argv)
    args.output_root = args.output_root.resolve()
    if args.output_root == args.track_dir.resolve() or args.track_dir.resolve() in args.output_root.parents:
        parser.error('diagnostics must be outside input track')
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='diagnosis-', dir=args.output_root))
    final, odom = None, None
    if args.export_graph:
        subprocess.run(['ros2','run','slam_graph_tools','export_slam_graph',str(args.track_dir.resolve()/'map'),str(out/'graph')], check=True)
        final, odom = out/'graph/trajectory.csv', out/'graph/odometry.csv'
    online = args.track_dir/'trajectory_online.csv'
    if not online.exists(): online = args.track_dir/'trajectory.csv'
    report = diagnose(args.track_dir/'map.yaml', online, final, odom, args.mapping_log, args.follower_log)
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    summary = concise(report)
    (out/'summary.txt').write_text(summary+'\n')
    print(summary + f'\nReport: {out}')
    return 0


if __name__ == '__main__':
    main()

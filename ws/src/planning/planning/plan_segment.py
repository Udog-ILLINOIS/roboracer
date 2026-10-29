#!/usr/bin/env python3
"""ros2 run planning plan_segment <name> --start-s S [--opponent GAP D V] [--mode A B C]

Plans minimum-time overtaking segments offline on a mapped track and reports how long they take to solve,
for trying window ends and margins before the planner runs live. The car starts on the global raceline at
arc length S (offset --start-d, speed --start-v, default the raceline's); the opponent (optional) is GAP m
further along the raceline (negative: behind, |GAP| under a lap), at offset D from it, driving V m/s along it.

Which raceline the segments start from and return to is reported (its build_raceline provenance line): one
that is not a clean converged TUM mintime line, or of unknown provenance, ends the output in a WARNING block
and is named in the PNG titles.

Reads   <tracks>/<name>/centerline.csv, raceline.csv, map.yaml (+ map.pgm)
Writes  <tracks>/<name>/segments/<mode>_<side>.csv (raceline columns, s from the car) and <mode>.png

--start-v and the opponent's V are real speeds; with --speed-scale (pure pursuit's speed_scale) both are
divided by it, since the raceline and the segments are planned at full speed and driven at speed_scale.
"""
import argparse
import json
import os
import sys

import numpy as np

from planning.build_raceline import default_config
from planning.raceline import TUM_PYTHON, OptimizationError, read_provenance, read_raceline, write_raceline
from planning.segment import run_segment


def start_state(rl, s, d, v):
    """Car on the raceline at arc length s, offset d to its left, heading along it (v: planned speed)."""
    s = s % rl[-1, 0]
    x, y = np.interp(s, rl[:, 0], rl[:, 1]), np.interp(s, rl[:, 0], rl[:, 2])
    psi = np.interp(s, rl[:, 0], np.unwrap(rl[:, 3]))
    yaw = psi + np.pi / 2                          # TUM psi: 0 = north
    vx = np.interp(s, rl[:, 0], rl[:, 5])
    return {'x': float(x - d * np.sin(yaw)), 'y': float(y + d * np.cos(yaw)), 'yaw': float(yaw),
            'v': float(vx if v is None else v)}


def same_file(a, b):
    """Same path, or both readable with the same contents."""
    if os.path.realpath(a) == os.path.realpath(b):
        return True
    try:
        with open(a, 'rb') as fa, open(b, 'rb') as fb:
            return fa.read() == fb.read()
    except OSError:
        return False


def raceline_report(rl_csv, config):
    """Which raceline this is: (label for the report and the PNG titles, warning lines, notes). Warnings unless
    it is a clean converged TUM mintime line; notes where the segments' vehicle config differs from the one the
    raceline was solved with."""
    try:
        prov, bad = read_provenance(rl_csv), None
    except ValueError as e:          # a '# raceline:' line that is not JSON
        prov, bad = None, str(e)
    if prov is not None and not isinstance(prov, dict):
        prov, bad = None, f'not a JSON object: {prov!r}'
    if prov is None:
        return ('raceline of UNKNOWN provenance',
                [f'{rl_csv} has no build_raceline provenance line' + (f' (unreadable: {bad})' if bad else '')
                 + ' (built before it was recorded, or written by hand), so it may not be a converged mintime line'],
                [])
    method, changed = prov.get('method'), prov.get('changed') or {}
    notes = []
    # changed keys other than path / speeds are racecar.ini settings the raceline was solved with (mintime-retry)
    ini = {k: v for k, v in changed.items() if k not in ('path', 'speeds')}
    if ini:
        notes.append(f'the raceline was solved with {json.dumps(ini)}, the segments with {config} as is')
    if prov.get('config') and not same_file(prov['config'], config):
        notes.append(f'the raceline was solved with {prov["config"]}, the segments with {config}')
    if method == 'mintime' and prov.get('clean'):
        return f'raceline TUM mintime, lap {prov.get("lap_s")} s', [], notes
    warn = ([f'method {method}, clean {prov.get("clean")}'] + [str(n) for n in prov.get('notes', [])]
            + ([f'differs from a mintime line: {json.dumps(changed)}'] if changed else []))
    return f'raceline {str(method).upper()} (NOT clean mintime)', warn, notes


def save_segment_preview(path, map_yaml, rl, resp, opp, title):
    """Map, raceline and each side's segment; with an opponent (dict s, d, v) its position at t = 0, its
    predicted path over the segments' time, and where it is when each side is closest to it."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from planning.centerline import OccupancyMap

    m = OccupancyMap(map_yaml)
    h, w = m.img.shape
    x0, y0 = m.origin[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * h / w + 1.0))
    ax.imshow(m.img, cmap='gray', vmin=0, vmax=255, extent=[x0, x0 + w * m.res, y0, y0 + h * m.res],
              interpolation='nearest')
    ax.plot(rl[:, 1], rl[:, 2], '-', color='0.5', lw=1, label='global raceline')
    pts = []
    for side, r in resp['sides'].items():
        if 'rows' in r:
            seg = np.asarray(r['rows'])
            pts.append(seg[:, 1:3])
            line, = ax.plot(seg[:, 1], seg[:, 2], '-', lw=2,
                            label='%s: %.3f s%s' % (side, r['segment_time'], '' if r['feasible'] else ' (infeasible)'))
            c = r.get('closest')
            if c:
                ax.plot(c['x'], c['y'], 'o', color=line.get_color(), ms=6)
                ax.plot(c['opp_x'], c['opp_y'], 's', mfc='none', mec=line.get_color(), mew=2, ms=9,
                        label='opponent at t = %.2f s, %s closest (%.2f m)' % (c['t'], side, c['dist']))
    if opp is not None:
        t_end = max([r['segment_time'] for r in resp['sides'].values() if 'segment_time' in r] or [0.0])
        o_path = np.array([opponent_xy(rl, opp, t) for t in np.linspace(0.0, t_end, 50)])
        if t_end > 0:
            ax.plot(o_path[:, 0], o_path[:, 1], '--', color='red', lw=1,
                    label='opponent predicted to t = %.2f s' % t_end)
        ax.plot(*o_path[0], 's', color='red', ms=8, label='opponent at t = 0')
        pts.append(o_path)
    if pts:
        p = np.vstack(pts)
        pad = 2.0
        ax.set_xlim(p[:, 0].min() - pad, p[:, 0].max() + pad)
        ax.set_ylim(p[:, 1].min() - pad, p[:, 1].max() + pad)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def opponent_xy(rl, opp, t):
    """The opponent's predicted (x, y) at time t: constant speed along the raceline, at its offset."""
    o = start_state(rl, opp['s'] + opp['v'] * t, opp['d'], None)
    return o['x'], o['y']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', help='track name, i.e. directory under the tracks dir')
    ap.add_argument('--tracks-dir', default=os.environ.get('TRACKS_DIR', os.path.expanduser('~/ws/tracks')))
    ap.add_argument('--config', default=default_config(), help='TUM racecar.ini (the one the raceline used)')
    ap.add_argument('--start-s', type=float, required=True, help='car position along the raceline [m]')
    ap.add_argument('--start-d', type=float, default=0.0, help='car offset left of the raceline [m]')
    ap.add_argument('--start-v', type=float, default=None,
                    help='car speed [m/s], default the raceline\'s planned speed there')
    ap.add_argument('--start-delta', type=float, default=None,
                    help='car steering angle [rad], default the raceline\'s there')
    ap.add_argument('--speed-scale', type=float, default=1.0,
                    help='pure pursuit speed_scale: --start-v and the opponent V are divided by it')
    ap.add_argument('--opponent', type=float, nargs=3, metavar=('GAP', 'D', 'V'), default=None,
                    help='opponent GAP m ahead along the raceline (negative: behind), D m left of it, V m/s along it')
    ap.add_argument('--opponent-size', type=float, nargs=2, metavar=('LENGTH', 'WIDTH'), default=(0.50, 0.27))
    ap.add_argument('--mode', nargs='+', choices=('A', 'B', 'C'), default=['A', 'B', 'C'])
    ap.add_argument('--kappa-min', type=float, default=0.2, help='corner threshold |kappa| [1/m]')
    ap.add_argument('--smooth', type=float, default=1.0, help='kappa is averaged over this for corners [m]')
    ap.add_argument('--min-turn', type=float, default=30.0, help='a corner turns at least this [deg]')
    ap.add_argument('--min-straight', type=float, default=1.0, help='shorter straights are driven through [m]')
    ap.add_argument('--corners-c', type=int, default=3, help='corners in a C window')
    ap.add_argument('--max-window', type=float, nargs=2, metavar=('LENGTH', 'LAP_FRACTION'), default=(60.0, 0.75),
                    help='windows are cut to LENGTH m and LAP_FRACTION of a lap')
    ap.add_argument('--margins', type=float, nargs=3, metavar=('M_S0', 'K_S', 'M_D'), default=None,
                    help='prediction margins: along the line at t = 0 [m], its growth [m/s], sideways [m]')
    ap.add_argument('--tum-python', default=TUM_PYTHON)
    args = ap.parse_args(argv)

    d = os.path.join(args.tracks_dir, args.name)
    cl, rl_csv = os.path.join(d, 'centerline.csv'), os.path.join(d, 'raceline.csv')
    for f in (cl, rl_csv):
        if not os.path.isfile(f):
            sys.exit(f'plan_segment: {f} not found (ros2 run planning build_raceline {args.name})')
    if args.speed_scale <= 0:
        sys.exit('plan_segment: --speed-scale must be > 0')
    rl = read_raceline(rl_csv)
    rl_label, rl_warn, rl_notes = raceline_report(rl_csv, args.config)
    print(f'{rl_label}: {rl_csv}' + ('' if rl_warn else ' (clean, converged)'))
    for n in rl_notes:
        print(f'NOTE: {n}')
    try:
        out_dir = os.path.join(d, 'segments')
        os.makedirs(out_dir, exist_ok=True)

        planned = (lambda v: v / args.speed_scale)  # noqa: E731
        start = start_state(rl, args.start_s, args.start_d, None if args.start_v is None else planned(args.start_v))
        start['delta'] = args.start_delta
        opp = None
        if args.opponent:
            gap, od, ov = args.opponent
            if abs(gap) >= rl[-1, 0]:
                sys.exit(f'plan_segment: --opponent GAP {gap} m: must be under a lap ({rl[-1, 0]:.1f} m) either way')
            # gap carried as is: tum_segment would otherwise take the copy of s nearest the car, so an opponent
            # more than half a lap ahead would be behind us
            opp = {'s': (args.start_s + gap) % rl[-1, 0], 'gap': gap, 'd': od, 'v': planned(ov),
                   'length': args.opponent_size[0], 'width': args.opponent_size[1]}

        print(f'{"mode":4s} {"side":5s} {"window":>8s} {"global":>8s} {"segment":>8s} {"iter":>5s} {"solve":>7s} '
              f'{"build":>7s} {"wall":>7s}  result')
        failed = False
        for mode in args.mode:
            req = {'centerline': cl, 'raceline': rl_csv, 'ini': args.config, 'start': start, 'opponent': opp,
                   'window': {'mode': mode, 'kappa_min': args.kappa_min, 'min_straight_m': args.min_straight,
                              'n_corners': args.corners_c, 'smooth_m': args.smooth, 'min_turn_deg': args.min_turn,
                              'max_length_m': args.max_window[0], 'max_lap_fraction': args.max_window[1]}}
            if args.margins:
                req['margins'] = dict(zip(('m_s0', 'k_s', 'm_d'), args.margins))
            try:
                resp, wall = run_segment(req, python=args.tum_python)
            except OptimizationError as e:
                print(f'{mode:4s} driver failed: {e}')
                failed = True
                continue
            win = resp['window']
            for side, r in resp['sides'].items():
                seg_t = '%8.3f' % r['segment_time'] if 'segment_time' in r else '%8s' % '-'
                res = 'feasible' if r['feasible'] else '; '.join(r['why'])
                if r.get('ahead_gap') is not None:
                    res += ', %.2f m ahead' % r['ahead_gap']
                if win.get('capped'):
                    res += ', window capped'
                if 'segment_time' in r and not r['settled']:
                    res += ', blocked stretch not settled'
                if r.get('reused_solves'):
                    res += ', %d solve(s) reused' % r['reused_solves']
                print(f'{mode:4s} {side:5s} {win["length_m"]:7.1f}m {resp["seed_time"]:7.3f}s {seg_t} '
                      f'{r["iterations"]:5d} {r["solve_time"]:6.2f}s {r["build_time"]:6.2f}s {wall:6.2f}s  {res}')
                if 'rows' in r:
                    write_raceline(os.path.join(out_dir, f'{mode}_{side}.csv'), np.asarray(r['rows']))
            if win.get('s_pass_rl') is not None:
                print(f'{mode:4s} pass done at raceline s = {win["s_pass_rl"] % rl[-1, 0]:.1f} m '
                      '(driving the raceline)')
            best = resp['best']
            title = (f'{args.name} {mode}: {win["length_m"]:.1f} m window, global {resp["seed_time"]:.3f} s, '
                     + (f'best {best} {resp["sides"][best]["segment_time"]:.3f} s' if best else 'no feasible segment')
                     + f'\n{rl_label}' + (' - WARNING' if rl_warn else ''))
            save_segment_preview(os.path.join(out_dir, f'{mode}.png'), os.path.join(d, 'map.yaml'), rl, resp, opp,
                                 title)
        print(f'-> {out_dir}   ({rl_label})')
    finally:
        # last whatever happens (a failed run, an exception), so a doubtful raceline is never missed
        if rl_warn:
            sys.stdout.flush()                   # the block comes last, also when piped
            bar = '!' * 100
            print('\n'.join([bar, f'WARNING: {rl_csv} is NOT a clean TUM mintime raceline: {rl_label}']
                            + [f'  - {w}' for w in rl_warn + rl_notes]
                            + ['  The segments start from and return to this line. Rebuild it: '
                               'ros2 run planning build_raceline ' + args.name, bar]), file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()

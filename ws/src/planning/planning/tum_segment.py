#!/usr/bin/env python3
"""Minimum-time overtaking segment. Runs in the TUM venv, NOT in ROS's Python.

  /opt/tum_venv/bin/python -I tum_segment.py <request.json> <response.json>

A segment is TUM's mintime problem (opt_mintime, same vehicle model and parameters as the global raceline) on an
open stretch of the reference line, from the car's state to a window end (segment_geometry.window_end, modes
A / B / C). The first state is fixed to the car's; the last is pulled towards the global raceline's state there
by a terminal cost; everything in between is free. The global raceline is only the seed (initial guess).
A window is at most max_length_m long and max_lap_fraction of a lap (then it ends short of its corner and the
response says capped).

The opponent (optional) is predicted at constant speed along the raceline, keeping its offset d from it. The
window ends are counted from where the pass is done: where we, driving the raceline at its speeds, would first
be ahead_gap_m (bumper to bumper) ahead of it; if that is not within the cap, both sides are rejected without
solving. Along the line everything is in raceline metres (our points by where their reference normals cross
the raceline), so car lengths and margins are real distances. Where we would be next to it (s within half
both lengths + m_s0 + k_s * t, t = our time there), its band [n - w/2 - m_d, n + w/2 + m_d] is removed from
the track, once with us passing on its left and once on its right; a side with no room left anywhere in its
blocked stretch is rejected without solving. Our timing changes with the solution, so the blocked stretch is
recomputed and the segment re-solved until it stops growing (at most max_block_iter solves; not settled
then: settled false, and rejected if points are too close, newly beside it or inside the blocked stretch, whose
band was placed at the previous solve's timing). A problem already solved (both sides
while nothing is blocked) is reused, not solved again. Passing on the left/right is feasible when IPOPT
converges, the end state is within tolerance of the raceline's, the solution keeps clear of the predicted
opponent at its own timing, and we end at least ahead_gap_m ahead of the opponent.

Request (JSON; lengths m, angles rad, speeds m/s):
  centerline, raceline, ini           paths: centerline.csv, raceline.csv, racecar.ini
  start {x, y, yaw, v, delta, beta, omega_z}   car state in the map frame; delta = its steering angle now
                                      (the first steering change obeys the steering-rate limit from it);
                                      defaults: delta and omega_z the raceline's there, beta 0;
                                      v below min_start_v is raised to it (TUM's model divides by v)
  window {mode: A|B|C, kappa_min, min_straight_m, n_corners, smooth_m, min_turn_deg} or {s_end}  (raceline
                                      arc length, any lap: the end is its first copy ahead of the car, and one
                                      under 10 points ahead is an error, not the next lap); either with
                                      max_length_m, max_lap_fraction
  opponent {s, d, v, length, width, gap}   raceline arc length, offset (left +) and speed; null = no opponent.
                                      gap: raceline metres ahead of the car (negative: behind), however far; then
                                      s only places it in the response. Without gap, the copy of s nearest the
                                      car (so more than half a lap ahead reads as behind)
  margins {m_s0, k_s, m_d}            prediction margins
  end_weights [w_v, w_beta, w_omega_z, w_n, w_xi]   terminal cost weights (scaled states)
  tolerances {n, xi, v}               end state vs raceline for feasibility
  ahead_gap_m, max_block_iter, min_start_v, ipopt {option: value}   extra IPOPT options (e.g. ipopt.tol)
Response: window {..., capped, s_pass_rl}, per side {feasible, why, status, iterations, solve_time, build_time,
  solver_builds, block_iterations, reused_solves, settled (null: not solved), unblocked_points, segment_time,
  residual {n, xi, v}, ahead_gap,
  closest {t, dist, x, y, opp_x, opp_y} (centres, when we are nearest the predicted opponent),
  rows [s, x, y, psi, kappa, vx, ax], t, n}, best, seed_time.
"""
import copy
import json
import os
import sys
import tempfile
import time

TUM_DIR = os.environ.get('TUM_DIR', '/opt/tum_optimizer')
sys.path.insert(0, TUM_DIR)

import numpy as np  # noqa: E402
import trajectory_planning_helpers as tph  # noqa: E402
import opt_mintime_traj  # noqa: E402

# our modules (-I drops the script's own directory from sys.path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planning import segment_geometry as sg  # noqa: E402
from planning.tum_mintime import load_pars, prep_reftrack  # noqa: E402

DEFAULTS = {
    'window': {'mode': 'B', 'kappa_min': 0.2, 'min_straight_m': 1.0, 'n_corners': 3, 'smooth_m': 1.0,
               'min_turn_deg': 30.0, 'max_length_m': 60.0, 'max_lap_fraction': 0.75},
    'margins': {'m_s0': 0.5, 'k_s': 0.5, 'm_d': 0.15},
    'end_weights': [100.0, 0.0, 0.0, 100.0, 100.0],
    'tolerances': {'n': 0.05, 'xi': 0.05, 'v': 0.2},
    'ahead_gap_m': 0.5,
    'max_block_iter': 4,
    'min_start_v': 0.5,
    # tol 1e-4 (TUM's lap: 1e-7): ~10x fewer iterations on porto, segment time within 0.3 %
    'ipopt': {'ipopt.max_iter': 3000, 'ipopt.tol': 1e-4, 'ipopt.acceptable_tol': 1e-3},
}


class Track:
    """The reference line TUM optimizes on and where the global raceline crosses each of its normals."""

    def __init__(self, centerline_csv, raceline_csv, pars):
        self.pars = pars
        reftrack, normvec, coeffs_x, coeffs_y = prep_reftrack(centerline_csv, pars)
        self.xy = reftrack[:, :2]
        self.w_right, self.w_left = reftrack[:, 2], reftrack[:, 3]
        self.normvec = normvec                    # TUM's normals point right
        self.nl = -normvec
        self.yaw = np.arctan2(*sg.tangents(self.nl)[:, ::-1].T)
        self.h = pars['stepsize_opts']['stepsize_reg']
        self.m = len(self.xy)
        # curvature exactly as opt_mintime computes it for the closed lap
        lengths = tph.calc_spline_lengths.calc_spline_lengths(coeffs_x=coeffs_x, coeffs_y=coeffs_y)
        cc = pars['curv_calc_opts']
        self.kappa = tph.calc_head_curv_num.calc_head_curv_num(
            path=self.xy, el_lengths=lengths, is_closed=True,
            stepsize_curv_preview=cc['d_preview_curv'], stepsize_curv_review=cc['d_review_curv'],
            stepsize_psi_preview=cc['d_preview_head'], stepsize_psi_review=cc['d_review_head'])[1]

        traj = np.loadtxt(raceline_csv, comments='#', delimiter=';', ndmin=2)
        self.raceline = sg.raceline_lap(traj)
        rl, lap = self.raceline
        n, j = sg.cross_normals(self.xy, self.nl, rl[:, 1:3], max_n=float(np.max(reftrack[:, 2:4])) + 0.5)
        if np.isnan(n).any():
            raise ValueError('the raceline does not cross the reference normals at %d points' % np.isnan(n).sum())
        self.n_rl = n
        self.v_rl = sg.interp_closed(rl[:, 5], j)
        self.kappa_rl = sg.interp_closed(rl[:, 4], j)
        j0 = np.floor(j).astype(int) % len(rl)
        d = rl[(j0 + 1) % len(rl), 1:3] - rl[j0, 1:3]
        self.xi_rl = sg.wrap_angle(np.arctan2(d[:, 1], d[:, 0]) - self.yaw)
        # raceline arc length at each reference point, unwrapped, and the map between the two arc lengths
        s_rl = sg.unwrap_lap(sg.interp_closed_s(rl[:, 0], lap, j), lap)
        self.lap_rl, self.lap_ref = lap, self.m * self.h
        k = np.arange(-1, 4)
        self._rl = (s_rl[None, :] + k[:, None] * lap).ravel()
        self._ref = ((np.arange(self.m) * self.h)[None, :] + k[:, None] * self.lap_ref).ravel()
        self.s_rl = s_rl

    def ref_of_rl(self, s):
        """Reference arc length (unwrapped) of a raceline arc length (unwrapped)."""
        return np.interp(s, self._rl, self._ref)

    def rl_of_ref(self, s):
        return np.interp(s, self._ref, self._rl)

    def n_at(self, s_ref):
        """Raceline offset n at reference arc lengths (any lap)."""
        return np.interp(np.asarray(s_ref) % self.lap_ref, np.arange(self.m) * self.h, self.n_rl,
                         period=self.lap_ref)


def solve(track, idx, x0, delta0, x_guess, x_end, w_end, w_right, w_left, ipopt, work):
    pars = copy.deepcopy(track.pars)
    pars['optim_opts']['segment'] = {
        'kappa': track.kappa[idx], 'x0': x0, 'delta0': delta0, 'x_guess': x_guess, 'x_end': x_end, 'w_end': w_end,
        'ipopt': {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', **ipopt}}
    reftrack = np.column_stack((track.xy[idx], w_right, w_left))
    t0 = time.perf_counter()
    res = opt_mintime_traj.src.opt_mintime.opt_mintime(
        reftrack=reftrack, coeffs_x=None, coeffs_y=None, normvectors=track.normvec[idx], pars=pars,
        tpamap_path=None, tpadata_path=None, export_path=work, print_debug=False, plot_debug=False)
    res['build_time'] = time.perf_counter() - t0 - res['solve_time']
    return res


def rows_of(track, idx, res):
    """Segment as raceline.csv rows [s, x, y, psi, kappa, vx, ax] (psi: TUM convention, 0 = north)."""
    x = res['x']
    xy = track.xy[idx] + track.nl[idx] * x[:, 3:4]
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(
        path=xy, el_lengths=np.maximum(seg, 1e-6), is_closed=False)
    ax = np.insert(res['ax'], 0, res['ax'][0])   # opt_mintime's ax[k] is at point k + 1
    return np.column_stack((s, xy, psi, kappa, x[:, 0], ax))


def plan(req, work):
    for k, v in DEFAULTS.items():
        req[k] = {**v, **(req.get(k) or {})} if isinstance(v, dict) else req.get(k, v)
    pars = load_pars(req['ini'])
    track = Track(req['centerline'], req['raceline'], pars)
    h, width_opt = track.h, pars['optim_opts']['width_opt']

    # the car, snapped to the nearest reference point (at most h / 2 along the line away)
    st = req['start']
    s_car, n_car = sg.project_point(track.xy, np.arange(track.m) * h, (st['x'], st['y']))
    i0 = int(round(s_car / h)) % track.m
    v0 = max(st['v'], req['min_start_v'])
    # unmeasured yaw rate / steering: the raceline's (v * kappa, kinematic); a car held at 0 in a bend can make
    # IPOPT fail
    omega_z = st['omega_z'] if st.get('omega_z') is not None else v0 * track.kappa_rl[i0]
    wheelbase = pars['vehicle_params_mintime']['wheelbase']
    delta0 = st['delta'] if st.get('delta') is not None else float(np.arctan(wheelbase * track.kappa_rl[i0]))
    x0 = [v0, st.get('beta', 0.0), omega_z, n_car, float(sg.wrap_angle(st['yaw'] - track.yaw[i0]))]
    s_car_rl = track.s_rl[i0]

    win = req['window']
    seed_all = np.column_stack((track.v_rl, np.zeros(track.m), track.v_rl * track.kappa_rl, track.n_rl,
                                track.xi_rl))

    def nodes(n_points):
        """Reference indices, our raceline arc length and the raceline's timing over n_points from the car."""
        i = i0 + np.arange(n_points)
        j = i % track.m
        sd = seed_all[j]
        return j, track.rl_of_ref(i * h), sg.segment_time(h, sd[:, 3], sd[:, 4], sd[:, 0], track.kappa[j])

    # the longest window, in reference metres (what the solve time grows with)
    s_cap_ref, _ = sg.cap_window(0.0, np.inf, track.lap_ref, win['max_length_m'], win['max_lap_fraction'])
    n_cap = int(s_cap_ref / h) + 1
    veh = pars['veh_params']
    opp = req.get('opponent')
    s_pass_rl = None
    if opp:
        marg = req['margins']
        gap_s = (opp['length'] + veh['length']) / 2              # centre to centre along the line, touching
        clear_n = opp['width'] / 2 + marg['m_d'] + veh['width'] / 2   # centre to centre sideways, with m_d
        s_exact = float(track.rl_of_ref(s_car))       # the car, not snapped
        if opp.get('gap') is not None:
            s_opp = s_exact + float(opp['gap'])
        else:
            # nearest copy of the opponent's s to the car, so one just ahead is not a lap ahead
            s_opp = s_exact + (opp['s'] - s_exact + track.lap_rl / 2) % track.lap_rl - track.lap_rl / 2

        def s_opp_at(t):     # raceline arc length, unwrapped
            return s_opp + opp['v'] * np.asarray(t)

        def n_opp_at(t):     # the opponent's own offset at time t, not the raceline's where we are
            return track.n_at(track.ref_of_rl(s_opp_at(t))) + opp['d']

        def opp_xy_at(t):
            return np.column_stack(sg.raceline_point(track.raceline, s_opp_at(t), opp['d']))

        # where the pass is done if we drive the raceline (its speeds, not the car's)
        _, s_cap_nodes, t_cap_nodes = nodes(n_cap)
        k = sg.pass_point(s_cap_nodes, t_cap_nodes, s_opp_at, gap_s + req['ahead_gap_m'])
        s_from = None if k is None else float(s_cap_nodes[k])
        s_pass_rl = s_from
    else:
        s_from = s_car_rl
    capped = False
    if 's_end' in win:
        # its first copy ahead of the car: s_rl is unwrapped from where reference point 0's normal crosses the
        # raceline, so the car's can be a lap above the raceline's own s
        s_end_rl = s_car_rl + (float(win['s_end']) - s_car_rl) % track.lap_rl
    elif s_from is None:                            # no pass: the sides are rejected, the window is the cap
        s_end_rl, capped = float(track.rl_of_ref((i0 + n_cap - 1) * h)), True
    else:
        s_end_rl = sg.window_end(track.raceline, s_from, win['mode'], kappa_min=win['kappa_min'],
                                 min_straight_m=win['min_straight_m'], n_corners=win['n_corners'],
                                 smooth_m=win['smooth_m'], min_turn=np.radians(win['min_turn_deg']))
    i_end = int(np.ceil(track.ref_of_rl(s_end_rl) / h))
    length = i_end - i0 + 1
    if length > n_cap:
        length, capped = n_cap, True
    if length < 10:
        raise ValueError('segment of %d points (%.1f m): needs at least 10' % (length, length * h))
    idx, s_nodes, _ = nodes(length)                 # s_nodes: raceline metres
    s_end_rl = float(s_nodes[-1])                   # the last point's (the end is rounded up to one)

    seed = seed_all[idx]
    x_end = seed[-1].copy()
    w_end = req['end_weights']
    seed_time = float(sg.segment_time(h, seed[:, 3], seed[:, 4], seed[:, 0], track.kappa[idx])[-1])
    out = {'window': {'i0': i0, 'points': int(length), 'length_m': float((length - 1) * h),
                      's_start_rl': float(s_car_rl), 's_end_rl': float(s_end_rl),
                      'mode': win.get('mode') if 's_end' not in win else None, 'capped': capped,
                      's_pass_rl': s_pass_rl},
           'seed_time': seed_time, 'sides': {}}

    if opp:
        def too_close(x, t):
            """Points where the segment (states x, times t) is next to the opponent with less than clear_n."""
            near = sg.alongside(s_nodes, t, s_opp_at, gap_s, marg['m_s0'], marg['k_s'])
            near[0] = False                                    # the car's own start state
            sep = np.abs(x[:, 3] - n_opp_at(t))
            return near, near & (sep < clear_n - 1e-3), sep

    solved = {}

    def solve_once(guess, w_r, w_l):
        """solve(), or the result of the same problem solved before (and True)."""
        key = (np.asarray(guess).tobytes(), w_r.tobytes(), w_l.tobytes())
        if key not in solved:
            solved[key] = solve(track, idx, x0, delta0, guess, x_end, w_end, w_r, w_l, req['ipopt'], work)
            return solved[key], False
        return solved[key], True

    sides = ['left', 'right'] if opp else ['free']
    for side in sides:
        r = {'feasible': False, 'why': [], 'iterations': 0, 'solve_time': 0.0, 'build_time': 0.0,
             'solver_builds': 0, 'block_iterations': 0, 'reused_solves': 0, 'settled': True}
        out['sides'][side] = r
        if opp and s_from is None:
            r['why'].append('not %.2f m ahead of the opponent within %.1f m on the raceline'
                            % (req['ahead_gap_m'], (n_cap - 1) * h))
            continue
        guess, mask, res = seed, np.zeros(length, dtype=bool), None
        t_nodes = sg.segment_time(h, seed[:, 3], seed[:, 4], seed[:, 0], track.kappa[idx])
        w_r, w_l = track.w_right[idx], track.w_left[idx]
        r['settled'] = not opp
        for it in range(req['max_block_iter'] if opp else 1):
            if opp:
                mask = mask | sg.alongside(s_nodes, t_nodes, s_opp_at, gap_s, marg['m_s0'], marg['k_s'])
                # TUM keeps our centre width_opt / 2 inside every bound, so the band is clear_n less that
                w_r, w_l = sg.block_widths(track.w_right[idx], track.w_left[idx], mask, n_opp_at(t_nodes),
                                           clear_n - width_opt / 2, side)
                room = w_r + w_l - width_opt
                tight = mask & (room < 0)
                tight[0] = False                                  # the start point is the car's, not checked
                if tight.any():
                    k = int(np.argmax(tight))
                    r['why'].append('no room on the %s of the opponent (from raceline s = %.1f m, %d points)'
                                    % (side, s_nodes[k] % track.lap_rl, tight.sum()))
                    res = None
                    break
            res, reused = solve_once(guess, w_r, w_l)
            r['block_iterations'] = it + 1
            r['reused_solves'] += int(reused)
            if not reused:
                r['iterations'] += int(res['iterations'])
                r['solve_time'] += res['solve_time']
                r['build_time'] += res['build_time']
                r['solver_builds'] += int(not res['cached'])
            if not res['success']:
                r['why'].append('IPOPT: %s' % res['status'])
                res = None
                break
            guess, t_nodes = res['x'], res['t']
            if opp:
                # settled: at its own timing the segment blocks nothing new and keeps clear of the opponent
                near, close, _ = too_close(res['x'], res['t'])
                if not (near & ~mask).any() and not close.any():
                    r['settled'] = True
                    break
        r['blocked_points'] = int(mask.sum())
        if not r['block_iterations']:
            r['settled'] = None                                 # rejected before any solve
        if res is not None:
            x = res['x']
            r['status'] = res['status']
            r['segment_time'] = float(res['t'][-1])
            r['residual'] = {'n': float(x[-1, 3] - x_end[3]), 'xi': float(sg.wrap_angle(x[-1, 4] - x_end[4])),
                             'v': float(x[-1, 0] - x_end[0])}
            tol = req['tolerances']
            for k in ('n', 'xi', 'v'):
                if abs(r['residual'][k]) > tol[k]:
                    r['why'].append('end %s off the raceline by %.3f (tolerance %.3f)'
                                    % (k, r['residual'][k], tol[k]))
            if opp:
                # against the solution's own timing (its blocked stretch came from the previous solve)
                near, close, sep = too_close(x, res['t'])
                grown = near & ~mask
                r['unblocked_points'] = int(grown.sum())   # next to the opponent but not blocked in the solve
                if close.any():
                    r['why'].append('passes %.2f m (centres) beside the opponent, needs %.2f'
                                    % (sep[close].min(), clear_n))
                    moved = close & mask
                    if not r['settled']:
                        parts = ([] if not grown.any() else ['%d points newly beside the opponent' % grown.sum()]) \
                            + ([] if not moved.any() else ['%d blocked points too close at this solve\'s timing '
                                                           '(band placed at the previous solve\'s)' % moved.sum()])
                        r['why'].append('not settled after %d solves: %s' % (r['block_iterations'], ', '.join(parts)))
                our_xy = track.xy[idx] + track.nl[idx] * x[:, 3:4]
                o_xy = opp_xy_at(res['t'])
                dist = np.hypot(*(our_xy - o_xy).T)
                k = int(np.argmin(dist))
                r['closest'] = {'t': float(res['t'][k]), 'dist': float(dist[k]), 'x': float(our_xy[k, 0]),
                                'y': float(our_xy[k, 1]), 'opp_x': float(o_xy[k, 0]), 'opp_y': float(o_xy[k, 1])}
                gap = s_nodes[-1] - s_opp_at(res['t'][-1]) - gap_s
                r['ahead_gap'] = float(gap)
                if gap < req['ahead_gap_m']:
                    r['why'].append('ends %.2f m ahead of the opponent (need %.2f)' % (gap, req['ahead_gap_m']))
            r['feasible'] = not r['why']
            r['rows'] = rows_of(track, idx, res).tolist()
            r['t'] = res['t'].tolist()
            r['n'] = x[:, 3].tolist()

    feasible = [(r['segment_time'], k) for k, r in out['sides'].items() if r['feasible']]
    out['best'] = min(feasible)[1] if feasible else None
    return out


def main():
    req_path, resp_path = sys.argv[1:3]
    with open(req_path) as f:
        req = json.load(f)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='segment_') as work:
        out = plan(req, work)
    out['total_time'] = time.perf_counter() - t0
    with open(resp_path, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()

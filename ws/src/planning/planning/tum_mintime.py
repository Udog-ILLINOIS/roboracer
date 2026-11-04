#!/usr/bin/env python3
"""TUM minimum-time optimization of one reference track. Runs in the TUM venv, NOT in ROS's Python.

  /opt/tum_venv/bin/python -I tum_mintime.py <centerline.csv> <racecar.ini> <traj_race_cl.csv> <work_dir>

Mirrors the opt_type = 'mintime' path of TUM's main_globaltraj.py (pinned commit, see
docker/install_deps.sh) with our inputs instead of its hard-coded ones:
  import track -> prep_track (spline regression; normals kept from crossing) -> opt_mintime (IPOPT)
  -> create_raceline -> heading/curvature -> mintime velocity profile -> export_traj_race.
No reoptimization and no ggv-based velocity recalculation: the velocity profile is the optimizer's own.

Exit code != 0 if IPOPT does not converge (opt_mintime exits 1).
"""
import configparser
import json
import os
import sys
import time

TUM_DIR = os.environ.get('TUM_DIR', '/opt/tum_optimizer')
sys.path.insert(0, TUM_DIR)

import numpy as np  # noqa: E402
import trajectory_planning_helpers as tph  # noqa: E402
import helper_funcs_glob  # noqa: E402
import opt_mintime_traj  # noqa: E402

# our numpy-only helpers (-I drops the script's own directory from sys.path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planning.track_geometry import uncross_normals  # noqa: E402


def load_pars(ini):
    parser = configparser.ConfigParser()
    if not parser.read(ini):
        raise IOError('cannot read ' + ini)
    g = lambda sec, key: json.loads(parser.get(sec, key))  # noqa: E731
    pars = {k: g('GENERAL_OPTIONS', k) for k in
            ('stepsize_opts', 'reg_smooth_opts', 'veh_params', 'vel_calc_opts', 'curv_calc_opts')}
    pars['optim_opts'] = g('OPTIMIZATION_OPTIONS', 'optim_opts_mintime')
    for k in ('vehicle_params_mintime', 'tire_params_mintime', 'pwr_params_mintime'):
        pars[k] = g('OPTIMIZATION_OPTIONS', k)
    pars['optim_opts']['var_friction'] = None
    pars['optim_opts']['warm_start'] = False
    v = pars['vehicle_params_mintime']
    v['wheelbase'] = v['wheelbase_front'] + v['wheelbase_rear']
    return pars


def prep_reftrack(track_csv, pars):
    """Reference line the optimizer works on: (reftrack [x, y, w_right, w_left], normvec (right), coeffs_x, coeffs_y).

    Deterministic, so tum_segment.py rebuilds exactly the line the global raceline was optimized on."""
    reftrack_imp = helper_funcs_glob.src.import_track.import_track(
        file_path=track_csv, width_veh=pars['veh_params']['width'],
        imp_opts={'flip_imp_track': False, 'set_new_start': False, 'num_laps': 1})

    # = helper_funcs_glob prep_track, except that widths are shrunk where the normals of the
    # smoothed line cross (prep_track only raises; the regression moves the line by a few cm)
    so, ro = pars['stepsize_opts'], pars['reg_smooth_opts']
    reftrack = tph.spline_approximation.spline_approximation(
        track=reftrack_imp, k_reg=ro['k_reg'], s_reg=ro['s_reg'], stepsize_prep=so['stepsize_prep'],
        stepsize_reg=so['stepsize_reg'], debug=True)
    coeffs_x, coeffs_y, a_interp, normvec = tph.calc_splines.calc_splines(
        path=np.vstack((reftrack[:, :2], reftrack[0, :2])))
    w_r, w_l = uncross_normals(reftrack[:, :2], -normvec, reftrack[:, 2], reftrack[:, 3],
                               horizon_m=12 * so['stepsize_reg'], margin=0.9)
    shrunk = np.sum((w_r < reftrack[:, 2]) | (w_l < reftrack[:, 3]))
    if shrunk:
        print('INFO: Narrowed the track at %d of %d points so its normals do not cross' % (shrunk, len(w_r)))
    reftrack[:, 2], reftrack[:, 3] = w_r, w_l
    if tph.check_normals_crossing.check_normals_crossing(track=reftrack, normvec_normalized=normvec, horizon=10):
        raise IOError('spline normals still cross after narrowing the track')
    return reftrack, normvec, coeffs_x, coeffs_y


def main():
    track_csv, ini, out_csv, work_dir = sys.argv[1:5]
    pars = load_pars(ini)
    os.makedirs(work_dir, exist_ok=True)
    t_start = time.perf_counter()

    reftrack, normvec, coeffs_x, coeffs_y = prep_reftrack(track_csv, pars)

    alpha_opt, v_opt, reftrack, a_tmp, normvec = opt_mintime_traj.src.opt_mintime.opt_mintime(
        reftrack=reftrack, coeffs_x=coeffs_x, coeffs_y=coeffs_y, normvectors=normvec, pars=pars,
        tpamap_path=None, tpadata_path=None, export_path=work_dir, print_debug=True,
        plot_debug=False)

    raceline, _, coeffs_x_opt, coeffs_y_opt, spline_inds, t_vals, s_points, spline_lengths, el_lengths = \
        tph.create_raceline.create_raceline(
            refline=reftrack[:, :2], normvectors=normvec, alpha=alpha_opt,
            stepsize_interp=pars['stepsize_opts']['stepsize_interp_after_opt'])

    # psi: TUM convention, 0 = +y (north), positive counter-clockwise, in [-pi, pi)
    psi, kappa = tph.calc_head_curv_an.calc_head_curv_an(
        coeffs_x=coeffs_x_opt, coeffs_y=coeffs_y_opt, ind_spls=spline_inds, t_spls=t_vals)

    s_splines = np.insert(np.cumsum(spline_lengths), 0, 0.0)
    vx = np.interp(s_points, s_splines[:-1], v_opt)
    ax = tph.calc_ax_profile.calc_ax_profile(vx_profile=np.append(vx, vx[0]), el_lengths=el_lengths,
                                             eq_length_output=False)
    t_lap = tph.calc_t_profile.calc_t_profile(vx_profile=vx, ax_profile=ax, el_lengths=el_lengths)[-1]
    print('INFO: Estimated laptime: %.2fs' % t_lap)

    traj = np.column_stack((s_points, raceline, psi, kappa, vx, ax))
    traj_cl = np.vstack((traj, traj[0]))
    traj_cl[-1, 0] = np.sum(spline_lengths)
    helper_funcs_glob.src.export_traj_race.export_traj_race(
        file_paths={'traj_race_export': out_csv}, traj_race=traj_cl)
    print('INFO: Runtime from import to final trajectory was %.2fs' % (time.perf_counter() - t_start))


if __name__ == '__main__':
    main()

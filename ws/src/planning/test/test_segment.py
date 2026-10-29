"""Overtaking segments on the fixture map: TUM mintime on an open window. The solver tests need the TUM venv."""
import json
import os
import shutil

import numpy as np
import pytest

from planning import build_raceline, plan_segment
from planning.raceline import HEADER, TUM_PYTHON, OptimizationError, read_raceline
from planning.segment import run_segment

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')
CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'racecar.ini')
needs_venv = pytest.mark.skipif(not os.path.isfile(TUM_PYTHON),
                                reason='TUM venv not installed: run the tests in the Docker container '
                                       '(docker compose exec ros, then python3 -m pytest src/planning/test)')


@pytest.fixture(scope='module')
def track(tmp_path_factory):
    """The fixture map with its centerline and global raceline (built once for this module)."""
    root = tmp_path_factory.mktemp('tracks')
    d = root / 'fixture'
    d.mkdir()
    shutil.copy(os.path.join(FIX, 'track_map_v3.pgm'), d / 'map.pgm')
    (d / 'map.yaml').write_text(open(os.path.join(FIX, 'track_map_v3.yaml')).read()
                                .replace('track_map_v3.pgm', 'map.pgm'))
    build_raceline.main(['fixture', '--tracks-dir', str(root), '--config', CONFIG])
    return root, d


def request(d, rl, s, mode='B', opponent=None, **window):
    return {'centerline': str(d / 'centerline.csv'), 'raceline': str(d / 'raceline.csv'), 'ini': CONFIG,
            'start': plan_segment.start_state(rl, s, 0.0, None), 'window': {'mode': mode, **window},
            'opponent': opponent}


@needs_venv
def test_free_segment_matches_the_global_line(track):
    _, d = track
    rl = read_raceline(d / 'raceline.csv')
    resp, _ = run_segment(request(d, rl, 2.0))
    r = resp['sides']['free']
    assert r['feasible'], r['why']
    # nothing in the way: the fastest segment is (about) the global line over the same window
    assert r['segment_time'] == pytest.approx(resp['seed_time'], rel=0.02)
    assert abs(r['residual']['n']) < 0.05 and abs(r['residual']['v']) < 0.2
    rows = np.asarray(r['rows'])
    assert rows.shape[1] == 7 and np.all(np.diff(rows[:, 0]) > 0)
    # starts at the car
    st = plan_segment.start_state(rl, 2.0, 0.0, None)
    assert np.hypot(rows[0, 1] - st['x'], rows[0, 2] - st['y']) < 0.15
    assert resp['window']['length_m'] > 1.0


@needs_venv
def test_opponent_sides_reported(track):
    _, d = track
    rl = read_raceline(d / 'raceline.csv')
    opp = {'s': 6.0, 'd': 0.0, 'v': 0.0, 'length': 0.5, 'width': 0.27}
    resp, _ = run_segment(request(d, rl, 2.0, 'A', opp))
    assert set(resp['sides']) == {'left', 'right'}
    for side, r in resp['sides'].items():
        assert r['feasible'] or r['why'], side
        if r['feasible']:
            assert r['ahead_gap'] >= 0.5
            assert r['blocked_points'] > 0
    if resp['best']:
        assert resp['sides'][resp['best']]['feasible']
    # one IPOPT solver for the window, reused by both sides and every re-solve
    assert sum(r['solver_builds'] for r in resp['sides'].values()) <= 1


@needs_venv
def test_opponent_behind_solved_once(track):
    _, d = track
    rl = read_raceline(d / 'raceline.csv')
    lap = rl[-1, 0]
    # 3 m behind us and slower: nothing is ever blocked, so both sides are the same problem
    opp = {'s': (2.0 - 3.0) % lap, 'd': 0.0, 'v': 1.0, 'length': 0.5, 'width': 0.27}
    resp, _ = run_segment(request(d, rl, 2.0, 'A', opp))
    left, right = resp['sides']['left'], resp['sides']['right']
    assert left['feasible'] and right['feasible'], (left['why'], right['why'])
    assert left['reused_solves'] == 0 and left['solve_time'] > 0
    assert right['reused_solves'] == right['block_iterations'] == 1 and right['solve_time'] == 0
    assert right['segment_time'] == left['segment_time'] and right['rows'] == left['rows']
    # the pass is done at the car: the window ends as without an opponent
    free, _ = run_segment(request(d, rl, 2.0, 'A'))
    assert resp['window']['s_end_rl'] == pytest.approx(free['window']['s_end_rl'])
    # ahead gap in raceline metres, by hand: window end - opponent there at the end - half both lengths
    s_opp = 2.0 - 3.0 + 1.0 * left['segment_time']
    hand = (resp['window']['s_end_rl'] - s_opp + lap / 2) % lap - lap / 2 - 0.5
    assert left['ahead_gap'] == pytest.approx(hand, abs=0.02)
    c = left['closest']
    assert c['t'] == 0.0 and c['dist'] == pytest.approx(3.0, abs=0.3)      # straight line, not along the track


@needs_venv
def test_opponent_out_of_reach_rejected_without_solving(track):
    _, d = track
    rl = read_raceline(d / 'raceline.csv')
    opp = {'s': 4.0, 'd': 0.0, 'v': 20.0, 'length': 0.5, 'width': 0.27}       # far faster than the raceline
    resp, _ = run_segment(request(d, rl, 2.0, 'A', opp))
    assert resp['window']['s_pass_rl'] is None
    for r in resp['sides'].values():
        assert not r['feasible'] and r['solve_time'] == 0 and 'rows' not in r
        assert 'ahead of the opponent within' in r['why'][0]


@needs_venv
def test_window_capped(track):
    _, d = track
    rl = read_raceline(d / 'raceline.csv')
    resp, _ = run_segment(request(d, rl, 2.0, 'C', max_length_m=3.0))
    w = resp['window']
    assert w['capped'] and w['length_m'] <= 3.0 + 1e-9
    resp, _ = run_segment(request(d, rl, 2.0, 'C', max_lap_fraction=0.2))
    assert resp['window']['capped']


@needs_venv
def test_unsettled_block_is_reported(track):
    _, d = track
    rl = read_raceline(d / 'raceline.csv')
    opp = {'s': 6.0, 'd': 0.0, 'v': 0.0, 'length': 0.5, 'width': 0.27}
    req = request(d, rl, 2.0, 'A', opp)
    req['max_block_iter'] = 1                   # no re-solve: the blocked stretch from the seed's timing only
    resp, _ = run_segment(req)
    for side, r in resp['sides'].items():
        if 'rows' not in r:
            continue
        assert r['block_iterations'] == 1
        # unsettled: new points next to the opponent were not blocked; rejected if they are too close
        if not r['settled']:
            close = any(w.startswith('passes') for w in r['why'])
            assert r['unblocked_points'] > 0 or close, side
            assert close == any(w.startswith('not settled after') for w in r['why']), side
        else:
            assert r['unblocked_points'] == 0


def rolled(rl, k):
    """The closed raceline started k points later (s from 0 there again)."""
    r = np.roll(rl[:-1], -k, axis=0)
    seg = np.hypot(*np.diff(np.vstack((r[:, 1:3], r[:1, 1:3])), axis=0).T)
    r[:, 0] = np.concatenate(([0.0], np.cumsum(seg)[:-1]))
    r = np.vstack((r, r[0]))
    r[-1, 0] = seg.sum()
    return r


@needs_venv
def test_window_end_given_on_a_raceline_starting_after_the_reference_start(track, tmp_path):
    from planning.raceline import write_raceline
    _, d = track
    # starts just after where reference point 0's normal crosses it: there the raceline's s is almost a lap
    rl = rolled(read_raceline(d / 'raceline.csv'), 3)
    write_raceline(tmp_path / 'raceline.csv', rl)
    for s_end in (6.0, 6.0 + rl[-1, 0]):                # any lap: the first copy ahead of the car
        req = request(d, rl, 2.0)
        req['raceline'], req['window'] = str(tmp_path / 'raceline.csv'), {'s_end': s_end}
        resp, _ = run_segment(req)
        w = resp['window']
        assert w['s_end_rl'] - w['s_start_rl'] == pytest.approx(4.0, abs=0.3)
        assert resp['sides']['free']['feasible'], resp['sides']['free']['why']


@needs_venv
def test_opponent_more_than_half_a_lap_ahead_stays_ahead(track):
    _, d = track
    rl = read_raceline(d / 'raceline.csv')
    lap = rl[-1, 0]
    # standing 0.6 laps ahead: by the nearest copy of its s it would be 0.4 laps behind, passed at the car;
    # ahead, the pass is past half a lap, beyond the window
    opp = {'s': (2.0 + 0.6 * lap) % lap, 'gap': 0.6 * lap, 'd': 0.0, 'v': 0.0, 'length': 0.5, 'width': 0.27}
    resp, _ = run_segment(request(d, rl, 2.0, 'A', opp, max_lap_fraction=0.5))
    assert resp['window']['s_pass_rl'] is None
    for r in resp['sides'].values():
        assert not r['feasible'] and 'rows' not in r and 'ahead of the opponent within' in r['why'][0]


# runs in the TUM venv: tum_segment.plan with a stand-in solver that keeps to the blocked widths but takes
# SLOW times as long as the raceline, so the opponent is further on when we reach it than the blocked stretch
# (from the raceline's timing) assumed
CLOSE_CHECK = '''
import json, sys, tempfile, numpy as np
sys.path.insert(0, '/opt/tum_optimizer'); sys.path.append(sys.argv[1])
from planning import tum_segment, segment_geometry as sg
SLOW = float(sys.argv[3])

def fake_solve(track, idx, x0, delta0, x_guess, x_end, w_end, w_right, w_left, ipopt, work):
    half = track.pars['optim_opts']['width_opt'] / 2
    x = np.array(x_guess, dtype=float)
    x[:, 3] = np.clip(x[:, 3], -w_right + half, w_left - half)
    t = SLOW * sg.segment_time(track.h, x[:, 3], x[:, 4], x[:, 0], track.kappa[idx])
    return {'success': True, 'status': 'fake', 'iterations': 1, 'solve_time': 0.0, 'build_time': 0.0,
            'cached': True, 'x': x, 't': t, 'ax': np.zeros(len(x) - 1)}

tum_segment.solve = fake_solve
with open(sys.argv[2]) as f:
    req = json.load(f)
print('RESULT ' + json.dumps(tum_segment.plan(req, tempfile.mkdtemp())))
'''


def close_check(d, rl, tmp_path, opp, slow, **req):
    import json
    import subprocess
    script, req_json = tmp_path / 'close.py', tmp_path / 'req.json'
    script.write_text(CLOSE_CHECK)
    req_json.write_text(json.dumps({**request(d, rl, 2.0, 'A', opp), **req}))
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([TUM_PYTHON, '-I', str(script), src, str(req_json), str(slow)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert out.returncode == 0, out.stdout[-2000:]
    return json.loads(out.stdout.split('RESULT ', 1)[1])


@needs_venv
def test_segment_beside_the_opponent_rejected_by_the_final_check(track, tmp_path):
    _, d = track
    rl = read_raceline(d / 'raceline.csv')
    opp = {'s': 6.0, 'd': 0.0, 'v': 1.0, 'length': 0.5, 'width': 0.27}
    # at the raceline's timing the right side settles and keeps clear (the left has no room on this map)
    resp = close_check(d, rl, tmp_path, opp, 1.0)
    r = resp['sides']['right']
    assert r['feasible'] and r['settled'], r['why']
    assert 'no room on the left' in resp['sides']['left']['why'][0]
    # twice as slow, it reaches the opponent later and further on, where nothing was blocked: rejected by the
    # final check (solved, past the pass point, inside the track), not before solving
    r = close_check(d, rl, tmp_path, opp, 2.0, max_block_iter=1)['sides']['right']
    assert not r['feasible'] and 'rows' in r and r['block_iterations'] == 1
    assert any(w.startswith('passes') and 'beside the opponent' in w for w in r['why']), r['why']
    assert any(w.startswith('not settled after 1 solves: ') and 'newly beside the opponent' in w
               for w in r['why']), r['why']
    assert r['unblocked_points'] > 0 and not r['settled']


@needs_venv
def test_standing_start_and_steering(track):
    _, d = track
    rl = read_raceline(d / 'raceline.csv')
    req = request(d, rl, 2.0, 'A')
    req['start']['v'] = 0.0                     # stopped: raised to min_start_v, TUM's model divides by v
    req['start']['delta'] = 0.0
    resp, _ = run_segment(req)
    r = resp['sides']['free']
    assert r['feasible'], r['why']
    assert np.asarray(r['rows'])[0, 5] == pytest.approx(0.5)


# runs in the TUM venv: the solver's own steering, which the segment response leaves out
STEER_CHECK = '''
import sys, tempfile, numpy as np
sys.path.insert(0, '/opt/tum_optimizer'); sys.path.append(sys.argv[1])
from planning.tum_segment import Track, solve
from planning.tum_mintime import load_pars
pars = load_pars(sys.argv[2])
t = Track(sys.argv[3], sys.argv[4], pars)
vm = pars['vehicle_params_mintime']
limit = vm['delta_max'] / vm['t_delta']
idx = (10 + np.arange(40)) % t.m
seed = np.column_stack((t.v_rl, np.zeros(t.m), t.v_rl * t.kappa_rl, t.n_rl, t.xi_rl))[idx]
for d0 in (0.15, -0.15):
    r = solve(t, idx, list(seed[0]), d0, seed, seed[-1], [100, 0, 0, 100, 100], t.w_right[idx], t.w_left[idx],
              {'ipopt.tol': 1e-4}, tempfile.mkdtemp())
    assert r['success'], r['status']
    first = abs(r['u'][0, 0] - d0) / (r['t'][1] - r['t'][0])
    print('RATE %.3f %.3f' % (first, limit))
'''


@needs_venv
def test_steering_rate_limited_from_the_cars_steering(track, tmp_path):
    import subprocess
    _, d = track
    script = tmp_path / 'steer.py'
    script.write_text(STEER_CHECK)
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([TUM_PYTHON, '-I', str(script), src, CONFIG, str(d / 'centerline.csv'),
                          str(d / 'raceline.csv')], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert out.returncode == 0, out.stdout[-2000:]
    rates = [tuple(map(float, line.split()[1:])) for line in out.stdout.splitlines() if line.startswith('RATE')]
    assert len(rates) == 2
    for first, limit in rates:
        # 0.15 rad off the line's steering: the first change is held to the limit, not taken at once
        assert first <= limit * 1.01


@needs_venv
def test_plan_segment_cli(track, capsys):
    root, d = track
    plan_segment.main(['fixture', '--tracks-dir', str(root), '--config', CONFIG, '--start-s', '2', '--mode', 'A'])
    out = capsys.readouterr().out
    assert 'A    free' in out
    assert (d / 'segments' / 'A.png').is_file()
    assert (d / 'segments' / 'A_free.csv').is_file()


def test_driver_failure_reported(tmp_path):
    fake = tmp_path / 'python'
    fake.write_text('#!/bin/sh\necho "Traceback: boom"\nexit 1\n')
    fake.chmod(0o755)
    with pytest.raises(OptimizationError, match='boom'):
        run_segment({}, python=str(fake))
    # IPOPT's exit and the one-line summary survive the re-wrap
    fake.write_text('#!/bin/sh\necho "EXIT: Maximum Number of Iterations Exceeded."\nexit 1\n')
    with pytest.raises(OptimizationError) as e:
        run_segment({}, python=str(fake))
    assert e.value.ipopt_exit == 'Maximum Number of Iterations Exceeded.'
    assert e.value.summary == 'segment IPOPT did not converge'


def test_opponent_xy_predicted():
    rl = np.array([[s, 1.0, s, 0.0, 0.0, 4.0, 0.0] for s in np.arange(0, 10.1, 0.5)])
    opp = {'s': 2.0, 'd': -0.5, 'v': 3.0}
    assert plan_segment.opponent_xy(rl, opp, 1.0) == pytest.approx((1.5, 5.0))   # right of north = east


def test_segment_preview_shows_the_predicted_opponent(tmp_path, monkeypatch):
    import matplotlib.axes
    rl = np.array([[s, 1.0, s, 0.0, 0.0, 4.0, 0.0] for s in np.arange(0, 10.1, 0.5)])
    (tmp_path / 'map.pgm').write_bytes(b'P5 20 20 255 ' + bytes([254]) * 400)
    (tmp_path / 'map.yaml').write_text('image: map.pgm\nresolution: 0.5\norigin: [-2.0, -1.0, 0.0]\n'
                                       'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    rows = rl[:9].copy()
    rows[:, 1] = 0.6
    resp = {'sides': {'left': {'rows': rows.tolist(), 'segment_time': 1.0, 'feasible': True,
                               'closest': {'t': 0.5, 'dist': 0.4, 'x': 0.6, 'y': 3.5, 'opp_x': 1.0, 'opp_y': 3.5}}}}
    labels = []
    orig = matplotlib.axes.Axes.plot

    def plot(self, *a, **kw):
        labels.append(kw.get('label'))
        return orig(self, *a, **kw)
    monkeypatch.setattr(matplotlib.axes.Axes, 'plot', plot)
    plan_segment.save_segment_preview(str(tmp_path / 'A.png'), str(tmp_path / 'map.yaml'), rl, resp,
                                      {'s': 2.0, 'd': 0.0, 'v': 1.5}, 'test')
    assert (tmp_path / 'A.png').is_file()
    assert 'opponent at t = 0' in labels
    assert 'opponent at t = 0.50 s, left closest (0.40 m)' in labels
    assert 'opponent predicted to t = 1.00 s' in labels


def test_start_state_on_raceline():
    # straight raceline heading north (TUM psi = 0) along x = 1
    rl = np.array([[s, 1.0, s, 0.0, 0.0, 4.0, 0.0] for s in np.arange(0, 10.1, 0.5)])
    st = plan_segment.start_state(rl, 3.0, 0.5, None)
    assert st['x'] == pytest.approx(0.5) and st['y'] == pytest.approx(3.0)   # left of north = west
    assert st['yaw'] == pytest.approx(np.pi / 2) and st['v'] == pytest.approx(4.0)
    assert plan_segment.start_state(rl, 3.0, 0.0, 2.5)['v'] == 2.5


def fake_track(tmp_path, provenance_line=None):
    """A straight 10 m raceline heading north on a blank map, for plan_segment without the TUM venv."""
    d = tmp_path / 'straight'
    d.mkdir()
    rl = np.array([[s, 1.0, s, 0.0, 0.0, 4.0, 0.0] for s in np.arange(0, 10.1, 0.5)])
    text = '# ' + HEADER + '\n' + ('' if provenance_line is None else '# ' + provenance_line + '\n')
    (d / 'raceline.csv').write_text(text + '\n'.join('; '.join('%.3f' % v for v in r) for r in rl) + '\n')
    (d / 'centerline.csv').write_text('# x_m; y_m; w_tr_right_m; w_tr_left_m\n')
    (d / 'map.pgm').write_bytes(b'P5 20 20 255 ' + bytes([254]) * 400)
    (d / 'map.yaml').write_text('image: map.pgm\nresolution: 0.5\norigin: [-2.0, -1.0, 0.0]\n'
                                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    return d, rl


def run_cli(monkeypatch, tmp_path, d, *extra):
    """plan_segment on a fake track with a canned response; returns (requests, PNG titles)."""
    reqs, titles = [], []
    rows = [[s, 1.0, s, 0.0, 0.0, 4.0, 0.0] for s in np.arange(0, 4.1, 0.5)]
    resp = {'window': {'length_m': 4.0, 'capped': False, 's_pass_rl': None}, 'seed_time': 1.0, 'best': 'free',
            'sides': {'free': {'feasible': True, 'why': [], 'rows': rows, 'segment_time': 1.0, 'iterations': 3,
                               'solve_time': 0.1, 'build_time': 0.1, 'settled': True}}}
    monkeypatch.setattr(plan_segment, 'run_segment', lambda req, python: (reqs.append(req) or resp, 0.2))
    monkeypatch.setattr(plan_segment, 'save_segment_preview', lambda *a: titles.append(a[-1]))
    plan_segment.main([d.name, '--tracks-dir', str(tmp_path), '--config', CONFIG, '--start-s', '2', '--mode', 'A',
                       *extra])
    return reqs, titles


def test_plan_segment_warns_about_a_raceline_of_unknown_provenance(monkeypatch, tmp_path, capsys):
    d, _ = fake_track(tmp_path)
    _, titles = run_cli(monkeypatch, tmp_path, d)
    out = capsys.readouterr()
    assert 'raceline of UNKNOWN provenance' in out.out and 'raceline of UNKNOWN provenance' in titles[0]
    assert 'WARNING: ' in out.err and 'NOT a clean TUM mintime raceline' in out.err
    assert 'no build_raceline provenance line' in out.err


def test_plan_segment_warns_about_a_fallback_raceline(monkeypatch, tmp_path, capsys):
    prov = {'method': 'mintime-retry', 'clean': False, 'notes': ['mintime failed: IPOPT did not converge'],
            'changed': {'safe_traj': False, 'speeds': 'capped'}, 'config': CONFIG}
    d, _ = fake_track(tmp_path, 'raceline: ' + json.dumps(prov))
    _, titles = run_cli(monkeypatch, tmp_path, d)
    out = capsys.readouterr()
    assert 'MINTIME-RETRY (NOT clean mintime)' in titles[0] and 'WARNING' in titles[0]
    assert 'NOT a clean TUM mintime raceline' in out.err and 'mintime failed: IPOPT did not converge' in out.err
    # the segments are solved with safe_traj on, the raceline was not
    assert 'NOTE: the raceline was solved with {"safe_traj": false}' in out.out


def test_plan_segment_clean_mintime_raceline_named(monkeypatch, tmp_path, capsys):
    prov = {'method': 'mintime', 'clean': True, 'notes': [], 'changed': {}, 'lap_s': 3.1, 'config': CONFIG}
    d, _ = fake_track(tmp_path, 'raceline: ' + json.dumps(prov))
    _, titles = run_cli(monkeypatch, tmp_path, d)
    out = capsys.readouterr()
    assert 'raceline TUM mintime, lap 3.1 s' in titles[0] and 'WARNING' not in titles[0]
    assert 'WARNING' not in out.err and 'NOTE' not in out.out
    assert 'raceline TUM mintime, lap 3.1 s' in out.out


def test_plan_segment_unparseable_provenance_warns(monkeypatch, tmp_path, capsys):
    d, _ = fake_track(tmp_path, 'raceline: {"method": mintime')
    run_cli(monkeypatch, tmp_path, d)
    assert 'unreadable' in capsys.readouterr().err


def test_plan_segment_opponent_gap_carried(monkeypatch, tmp_path):
    d, rl = fake_track(tmp_path)
    lap = rl[-1, 0]
    # more than half a lap ahead: still ahead, not (by the nearest copy of its s) behind the car
    reqs, _ = run_cli(monkeypatch, tmp_path, d, '--opponent', str(0.7 * lap), '0', '1')
    assert reqs[0]['opponent']['gap'] == pytest.approx(0.7 * lap)
    assert reqs[0]['opponent']['s'] == pytest.approx((2.0 + 0.7 * lap) % lap)
    with pytest.raises(SystemExit, match='under a lap'):
        run_cli(monkeypatch, tmp_path, d, '--opponent', str(lap), '0', '1')


def test_plan_segment_warning_survives_a_failed_run(monkeypatch, tmp_path, capsys):
    d, _ = fake_track(tmp_path)

    def boom(req, python):
        raise KeyError('window')
    monkeypatch.setattr(plan_segment, 'run_segment', boom)
    with pytest.raises(KeyError):
        plan_segment.main([d.name, '--tracks-dir', str(tmp_path), '--config', CONFIG, '--start-s', '2'])
    assert 'NOT a clean TUM mintime raceline' in capsys.readouterr().err
    # a bad --speed-scale is refused before anything is reported
    with pytest.raises(SystemExit, match='speed-scale'):
        plan_segment.main([d.name, '--tracks-dir', str(tmp_path), '--config', CONFIG, '--start-s', '2',
                           '--speed-scale', '0'])

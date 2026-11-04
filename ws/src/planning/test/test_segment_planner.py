"""segment_planner_node: a segment as offsets from the raceline, and the solver process."""
import json
import os
import subprocess

import numpy as np
import pytest

pytest.importorskip('rclpy')
from planning.segment_planner_node import DRIVER, plan_offsets  # noqa: E402
from planning.raceline import TUM_PYTHON  # noqa: E402
from test_segment import CONFIG, needs_venv, read_raceline, request, track  # noqa: E402,F401


def straight_project(lap):
    def project(points):
        p = np.reshape(points, (-1, 2))
        return p[:, 0] % lap, p[:, 1], np.abs(p[:, 1])
    return project


def test_offsets_follow_the_segment_and_are_zero_outside_it():
    lap = 100.0
    s_rail = np.arange(0.0, lap, 0.5)
    x = np.arange(10.0, 20.01, 0.4)
    rows = np.zeros((len(x), 7))
    rows[:, 1], rows[:, 2] = x, 0.3 * np.sin((x - 10.0) / 10.0 * np.pi)
    out = plan_offsets(straight_project(lap), s_rail, lap, rows)
    assert out[int(15 / 0.5)] == pytest.approx(0.3, abs=0.01)
    assert out[int(5 / 0.5)] == 0.0 and out[int(25 / 0.5)] == 0.0


def test_offsets_across_the_lap_line():
    lap = 30.0
    s_rail = np.arange(0.0, lap, 0.5)
    x = np.arange(28.0, 33.01, 0.4)
    rows = np.zeros((len(x), 7))
    rows[:, 1], rows[:, 2] = x, 0.2
    out = plan_offsets(straight_project(lap), s_rail, lap, rows)
    assert out[int(29 / 0.5)] == pytest.approx(0.2) and out[int(2 / 0.5)] == pytest.approx(0.2)
    assert out[int(10 / 0.5)] == 0.0


@needs_venv
def test_solver_process_keeps_serving(track):  # noqa: F811
    _, d = track
    rl = read_raceline(d / 'raceline.csv')
    req = {**request(d, rl, 2.0, 'A'), 'ini': CONFIG}
    proc = subprocess.Popen([TUM_PYTHON, '-I', DRIVER, '--serve'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, env=dict(os.environ))
    try:
        for _ in range(2):
            proc.stdin.write(json.dumps(req) + '\n')
            proc.stdin.flush()
            assert json.loads(proc.stdout.readline())['sides']['free']['feasible']
        proc.stdin.write(json.dumps({'centerline': 'missing'}) + '\n')    # a bad request: an error, not the end
        proc.stdin.flush()
        assert 'error' in json.loads(proc.stdout.readline())
    finally:
        proc.kill()

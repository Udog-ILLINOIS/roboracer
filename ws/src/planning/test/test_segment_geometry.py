"""Overtaking segment geometry on synthetic lines (numpy only, no TUM venv needed)."""
import numpy as np
import pytest

from planning import segment_geometry as sg


def circle(r, n=400):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack((r * np.cos(a), r * np.sin(a)))   # counter-clockwise: left is the centre
    nl = -xy / r
    return xy, nl, r * a


def raceline(kappa, ax, step=0.1):
    """Open raceline rows [s, x, y, psi, kappa, vx, ax] (only s, kappa and ax matter here) + lap."""
    n = len(kappa)
    rows = np.zeros((n, 7))
    rows[:, 0] = np.arange(n) * step
    rows[:, 4], rows[:, 6], rows[:, 5] = kappa, ax, 3.0
    return rows, n * step


def test_project_point_on_circle():
    xy, _, s = circle(2.0)
    s_p, d = sg.project_point(xy, s, (0.0, 1.5))           # quarter lap, 0.5 m inside = left
    assert s_p == pytest.approx(np.pi, abs=0.02)
    assert d == pytest.approx(0.5, abs=0.01)
    _, d_out = sg.project_point(xy, s, (0.0, 2.4))
    assert d_out == pytest.approx(-0.4, abs=0.01)


def test_cross_normals_concentric_circles():
    ref_xy, ref_nl, _ = circle(2.0)
    line_xy, _, _ = circle(1.7, n=300)
    n, j = sg.cross_normals(ref_xy, ref_nl, line_xy, max_n=1.0)
    assert np.allclose(n, 0.3, atol=2e-3)                   # 0.3 m to the left (inside)
    assert not np.isnan(j).any()
    n_far, _ = sg.cross_normals(ref_xy, ref_nl, line_xy, max_n=0.2)
    assert np.isnan(n_far).all()                            # out of reach


def test_corners_wrap_around():
    k = np.zeros(100)
    k[10:20] = 1.0
    k[95:] = -1.0
    k[:3] = -1.0                                            # one corner across the lap end
    assert sg.corners(k, 0.5) == [(10, 19), (95, 2)]
    assert sg.corners(np.zeros(5), 0.5) == []


def test_track_corners_join_noise_and_drop_bumps():
    ds = np.full(1000, 0.1)
    k = np.zeros(1000)
    k[200:250] = 1.0
    k[220:223] = 0.1                                        # noise in the corner: not two corners
    k[500:512] = 1.0                                        # a short corner: 1.2 m at 1 1/m turns 69 deg
    k[800:806] = 0.4                                        # a bump: 0.6 m at 0.4 1/m turns 14 deg
    assert len(sg.corners(k, 0.5)) == 3
    assert sg.track_corners(k, ds, 0.5) == [(200, 249), (500, 511)]
    assert sg.track_corners(k, ds, 0.5, min_turn=np.radians(90)) == [(200, 249)]
    assert sg.track_corners(k, ds, 0.2, min_turn=0.0)[-1] == (800, 805)
    assert (800, 805) not in sg.track_corners(k, ds, 0.2)                 # under min_turn
    # a chicane's two turns stay apart (signed kappa is averaged; it only moves their ends apart)
    k2 = np.zeros(1000)
    k2[200:250], k2[252:300] = 1.0, -1.0
    assert sg.track_corners(k2, ds, 0.5) == [(200, 247), (254, 299)]


def test_braking_point():
    ax = np.zeros(50)
    ax[20:30] = -3.0
    assert sg.braking_point(ax, 30) == 20
    assert sg.braking_point(ax, 10) == 10                   # no deceleration before it


def test_window_ends():
    # 100 m lap: corners at 20-25 m, 50-55 m and 80-85 m, each braked into over the 5 m before it
    k, ax = np.zeros(1000), np.zeros(1000)
    for c in (200, 500, 800):
        k[c:c + 50] = 1.0
        ax[c - 50:c] = -3.0
    rl = raceline(k, ax)
    km = {'kappa_min': 0.5}                 # half of the corners' kappa: the 1 m average keeps their ends
    assert sg.window_end(rl, 10.0, 'A', **km) == pytest.approx(24.9)
    assert sg.window_end(rl, 10.0, 'B', **km) == pytest.approx(45.0)        # braking into the 50 m corner
    assert sg.window_end(rl, 10.0, 'C', n_corners=3, **km) == pytest.approx(115.0)   # next lap's 20 m corner
    # from inside a corner: A is the exit of the next one
    assert sg.window_end(rl, 22.0, 'A', **km) == pytest.approx(54.9)
    # unwrapped: the second lap
    assert sg.window_end(rl, 110.0, 'A', **km) == pytest.approx(124.9)


def test_window_end_drives_through_a_chicane():
    k, ax = np.zeros(1000), np.zeros(1000)
    k[200:250] = 1.0
    k[255:300] = -1.0                       # 0.5 m straight: a chicane
    k[600:650] = 1.0
    ax[550:600] = -3.0
    rl = raceline(k, ax)
    assert sg.window_end(rl, 10.0, 'B', kappa_min=0.5, min_straight_m=1.0) == pytest.approx(55.0)
    assert sg.window_end(rl, 10.0, 'B', kappa_min=0.5, min_straight_m=0.2) == pytest.approx(25.5)


def test_window_end_errors():
    rl = raceline(np.zeros(100), np.zeros(100))
    with pytest.raises(ValueError, match='no corners'):
        sg.window_end(rl, 0.0, 'A')
    with pytest.raises(ValueError, match='mode'):
        sg.window_end(rl, 0.0, 'D')


def test_window_end_sweeping_corners():
    # 200 m lap of 3.3 m radius bends (kappa 0.3) with a noise spike to 0.6 in each: at kappa_min 0.5 there
    # are no corners, at the default 0.2 the bends are (their ends 0.2 m in: kappa averaged over 1 m)
    k, ax = np.zeros(2000), np.zeros(2000)
    for c in (300, 900, 1500):
        k[c:c + 80] = 0.3
        k[c + 40] = 0.6
    rl = raceline(k, ax)
    with pytest.raises(ValueError, match='no corners'):
        sg.window_end(rl, 10.0, 'A', kappa_min=0.5)
    assert sg.window_end(rl, 10.0, 'A') == pytest.approx(37.7)
    assert sg.window_end(rl, 10.0, 'B') == pytest.approx(90.2)


def test_cap_window():
    assert sg.cap_window(10.0, 30.0, 100.0, 60.0, 0.75) == (30.0, False)
    assert sg.cap_window(10.0, 90.0, 100.0, 60.0, 0.75) == (70.0, True)       # max length
    assert sg.cap_window(10.0, 40.0, 27.0, 60.0, 0.75) == (30.25, True)       # 3/4 of a short lap


def test_pass_point():
    s = np.arange(0, 20.0, 1.0)
    t = s / 5.0                                             # we drive 5 m/s
    # opponent 4 m ahead at 3 m/s: 1 m ahead of it at 2 m/s closing = 2.5 s, s = 12.5 -> point 13
    assert sg.pass_point(s, t, lambda tt: 4.0 + 3.0 * tt, 1.0) == 13
    assert sg.pass_point(s, t, lambda tt: 4.0 + 6.0 * tt, 1.0) is None       # faster: never
    assert sg.pass_point(s, t, lambda tt: -4.0 + 1.0 * tt, 1.0) == 0         # behind us already


def test_raceline_point():
    # raceline heading north (TUM psi = 0) along x = 1, 10 m lap
    rows = np.array([[s, 1.0, s, 0.0, 0.0, 4.0, 0.0] for s in np.arange(0, 10.0, 0.5)])
    x, y = sg.raceline_point((rows, 10.0), np.array([3.0, 13.25]), 0.5)
    assert np.allclose(x, 0.5) and np.allclose(y, [3.0, 3.25])                # left of north = west


def test_raceline_lap():
    t = np.array([[0.0, 0, 0, 0, 0, 1, 0], [1.0, 1, 0, 0, 0, 1, 0], [2.0, 0, 0, 0, 0, 1, 0]])
    rows, lap = sg.raceline_lap(t)
    assert len(rows) == 2 and lap == 2.0


def test_block_widths():
    w_r, w_l = np.ones(4), np.ones(4)
    mask = np.array([False, True, True, False])
    r, l_ = sg.block_widths(w_r, w_l, mask, 0.2, 0.3, 'left')
    assert np.allclose(r, [1, -0.5, -0.5, 1]) and np.allclose(l_, 1)
    r, l_ = sg.block_widths(w_r, w_l, mask, 0.2, 0.3, 'right')
    assert np.allclose(l_, [1, -0.1, -0.1, 1]) and np.allclose(r, 1)
    r, _ = sg.block_widths(w_r, w_l, mask, -5.0, 0.3, 'left')
    assert np.allclose(r, 1)                                # only ever narrower
    with pytest.raises(ValueError):
        sg.block_widths(w_r, w_l, mask, 0.0, 0.3, 'up')


def test_alongside_static_and_moving():
    s = np.arange(0, 20.0, 1.0)
    t = s / 5.0                                             # we drive 5 m/s
    static = sg.alongside(s, t, lambda tt: np.full_like(tt, 10.0), 0.5, 0.5, 0.0)
    assert np.flatnonzero(static).tolist() == [9, 10, 11]
    # opponent 4 m ahead at 3 m/s: we meet it at t = 2 s, s = 10 m, and stay next to it for longer
    moving = sg.alongside(s, t, lambda tt: 4.0 + 3.0 * tt, 0.5, 0.5, 0.0)
    assert np.flatnonzero(moving).tolist() == [8, 9, 10, 11, 12]
    grown = sg.alongside(s, t, lambda tt: 4.0 + 3.0 * tt, 0.5, 0.5, 1.0)
    assert moving.sum() < grown.sum() and grown[10]


def test_segment_time_straight():
    t = sg.segment_time(0.2, np.zeros(11), np.zeros(11), np.full(11, 4.0), np.zeros(11))
    assert t[-1] == pytest.approx(0.5)
    # offset to the inside of a bend shortens the path
    t_in = sg.segment_time(0.2, np.full(11, 0.5), np.zeros(11), np.full(11, 4.0), np.full(11, 0.5))
    assert t_in[-1] == pytest.approx(0.5 * 0.75)


def test_unwrap_lap():
    s = sg.unwrap_lap([9.0, 9.5, 0.2, 0.8, 1.5], 10.0)        # starts just before the lap end
    assert s == pytest.approx([9.0, 9.5, 10.2, 10.8, 11.5])
    # a normal reaching across a thin wall to another part of the line: refused, not interpolated
    with pytest.raises(ValueError, match='out of order: from reference point 2 to 3'):
        sg.unwrap_lap([1.0, 2.0, 6.5, 3.0, 4.0], 10.0)
    with pytest.raises(ValueError, match='out of order'):
        sg.unwrap_lap([1.0, 2.0, 2.0, 3.0], 10.0)
    # steps of more than half a lap either way pass the step check, but then the points span more than a lap
    # (e.g. a thin-wall hairpin straddling s = 0)
    for i, bad in ((10, 70.0), (80, 20.0)):
        s = np.arange(0.0, 100.0)
        s[i] = bad
        with pytest.raises(ValueError, match=r'span 1\d\d\.\d\d m of a 100\.00 m lap'):
            sg.unwrap_lap(s, 100.0)
    assert sg.unwrap_lap(np.arange(0.0, 100.0), 100.0)[-1] == 99.0     # a whole lap, closing step forward

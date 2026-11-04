"""planning.pass_line: the box QP, the pass line's shape, and the comparison of lines by time."""
import numpy as np

from planning.pass_line import box_qp, path_curvature, solve_offsets, speed_profile, time_lost


def pentadiagonal(n, seed=0):
    """A random symmetric positive definite pentadiagonal matrix: (dense, diag, off1, off2)."""
    rng = np.random.default_rng(seed)
    off1, off2 = rng.normal(size=n - 1), rng.normal(size=n - 2)
    diag = 2 * (np.abs(np.r_[off1, 0]) + np.abs(np.r_[0, off1]) + np.abs(np.r_[off2, 0, 0]) + np.abs(np.r_[0, 0, off2])) + 1
    dense = np.diag(diag) + np.diag(off1, 1) + np.diag(off1, -1) + np.diag(off2, 2) + np.diag(off2, -2)
    return dense, diag, off1, off2


def test_box_qp_meets_the_optimality_conditions():
    for seed in range(5):
        n = 30
        h, diag, off1, off2 = pentadiagonal(n, seed)
        rng = np.random.default_rng(seed + 100)
        g = rng.normal(size=n) * 5
        lower, upper = -rng.uniform(0.1, 1, n), rng.uniform(0.1, 1, n)
        lower[7], upper[7] = 0.3, 0.3    # a fixed point
        upper[3] = np.inf                # one-sided
        x = box_qp(diag, off1, off2, g, lower, upper)
        assert x is not None
        assert np.all(x >= lower - 1e-9) and np.all(x <= upper + 1e-9)
        assert abs(x[7] - 0.3) < 1e-9
        grad = h @ x + g
        interior = (x > lower + 1e-4) & (x < upper - 1e-4)
        assert np.max(np.abs(grad[interior]), initial=0.0) < 1e-4
        assert np.all(grad[(x <= lower + 1e-4) & (lower < upper)] >= -1e-4)   # pushed against the lower bound
        assert np.all(grad[(x >= upper - 1e-4) & (lower < upper)] <= 1e-4)    # and the upper


def test_box_qp_refuses_crossed_bounds():
    _, diag, off1, off2 = pentadiagonal(10)
    lower, upper = -np.ones(10), np.ones(10)
    lower[4], upper[4] = 0.5, 0.2
    assert box_qp(diag, off1, off2, np.zeros(10), lower, upper) is None


def straight(n=150, v=4.0):
    return np.zeros(n), np.full(n, v)


def around(lower, upper, near, far, goal, ds=0.1):
    r = ds * np.arange(len(lower))
    lower[(r >= near) & (r <= far)] = goal
    return lower, upper


def test_passes_at_the_goal_and_rejoins():
    kappa, v = straight()
    n = len(kappa)
    lower, upper = around(np.full(n, -1.0), np.full(n, 1.0), 4.0, 5.0, 0.5)
    lower[0] = upper[0] = 0.0
    d = solve_offsets(0.1, kappa, v, lower, upper)
    assert d is not None
    assert d[0] == 0.0 and abs(d[-1]) < 1e-9 and abs(d[-2]) < 1e-9
    assert np.all(d[40:51] >= 0.5 - 1e-6)
    assert np.max(np.abs(np.diff(d, 2))) / 0.01 < 1.0   # no kink


def test_the_return_is_as_long_as_it_is_given():
    # on a straight nothing else shapes it: the further away the end, the gentler the return
    kappa, v = straight(200)
    n = len(kappa)
    curvatures = []
    for end in (80, 120, 200):
        lower, upper = around(np.full(n, -1.0)[:end], np.full(n, 1.0)[:end], 4.0, 5.0, 0.5)
        lower[0] = upper[0] = 0.0
        d = solve_offsets(0.1, kappa[:end], v[:end], lower, upper)
        curvatures.append(np.max(np.abs(np.diff(d[50:], 2))) / 0.01)
    assert curvatures[0] > curvatures[1] > curvatures[2]


def test_starts_where_the_car_is_heading():
    kappa, v = straight()
    n = len(kappa)
    lower, upper = np.full(n, -1.0), np.full(n, 1.0)
    lower[0] = upper[0] = 0.2
    d = solve_offsets(0.1, kappa, v, lower, upper, slope0=0.1)
    assert abs(d[0] - 0.2) < 1e-9 and abs(d[1] - d[0] - 0.1 * 0.1) < 1e-9


def test_no_room_is_no_line():
    kappa, v = straight()
    n = len(kappa)
    lower, upper = around(np.full(n, -1.0), np.full(n, 0.3), 4.0, 5.0, 0.5)   # the goal is beyond the wall
    lower[0] = upper[0] = 0.0
    assert solve_offsets(0.1, kappa, v, lower, upper) is None


def test_speed_profile_limits():
    n = 200
    kappa = np.zeros((1, n))
    kappa[0, 100:120] = 0.5
    v = speed_profile(np.full((1, n - 1), 0.1), kappa, 10.0, 7.0, 3.0, 4.0, np.array([2.0]))[0]
    assert v[0] == 2.0
    assert np.isclose(v[110], np.sqrt(7.0 / 0.5))
    step = np.diff(v ** 2) / (2 * 0.1)
    assert step.max() < 3.0 + 1e-9 and step.min() > -4.0 - 1e-9


def test_time_lost_prefers_the_gentle_line_and_counts_none_for_the_reference():
    x = np.arange(0.0, 20.0, 0.1)
    reference = np.c_[x, np.zeros_like(x)]
    sharp, gentle = np.zeros_like(x), np.zeros_like(x)
    sharp[(x > 8) & (x < 10)] = np.sin(np.pi * (x[(x > 8) & (x < 10)] - 8) / 2) * 0.5
    gentle[(x > 2) & (x < 18)] = np.sin(np.pi * (x[(x > 2) & (x < 18)] - 2) / 16) * 0.5
    paths = np.stack([reference, reference + np.c_[0 * x, sharp], reference + np.c_[0 * x, gentle]])
    lost, _, _ = time_lost(paths, reference, 5.0, 8.0, 7.0, 3.0, 4.0)
    assert abs(lost[0]) < 1e-9
    assert lost[1] > lost[2] > 0.0


def test_path_curvature_of_a_circle():
    t = np.linspace(0, 2, 100)
    circle = 4.0 * np.c_[np.cos(t), np.sin(t)]
    kappa = path_curvature(circle[None])[1][0]
    assert np.allclose(kappa[3:-3], 0.25, atol=1e-3)

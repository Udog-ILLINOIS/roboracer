"""The pass line: the offset from the raceline that goes round an obstacle and rejoins it, solved in about a millisecond.

The raceline is a minimum-time line, so a small sideways offset d(s) costs time only to second order, and how much
depends on where along the lap it is. That makes the offset a small convex problem (a QP) in the raceline's frame,
on its points from the car to rejoin:

    minimize  sum grip_i * (d'' + kappa^2 d)^2     the added curvature, as lateral acceleration at the planned speed
            + sum length_i * d'^2                  the added path length, as time at the planned speed
    subject to  lower_i <= d_i <= upper_i          the track, and the obstacle cut out of it
                d = the car's at the start, d = d' = 0 at the end

The shape is not prescribed: the rise is as long as the distance to the obstacle allows, and the return is as long as
it is cheap, ending where the track closes in on the raceline. Which end point loses least is decided by time_lost,
a quasi-steady-state speed profile of the whole candidate.

The Hessian is pentadiagonal, so each interior-point step is one banded Cholesky factorization: numpy and scipy only.
"""
import numpy as np
from scipy.linalg import LinAlgError, solveh_banded


def box_qp(diag, off1, off2, linear, lower, upper, iterations=40, tol=1e-6):
    """Minimize 1/2 x'Hx + linear'x with lower <= x <= upper, H symmetric positive definite and pentadiagonal (its
    diagonal, first and second off-diagonal), by Mehrotra's predictor-corrector. A point with lower == upper is fixed.
    Returns x, or None if the bounds cross or the solver does not converge."""
    n = len(diag)
    if np.any(lower > upper):
        return None
    fixed = upper <= lower + 1e-9
    d0, d1, d2, g = diag.copy(), off1.copy(), off2.copy(), linear.copy()
    x_fix = np.zeros(n)
    x_fix[fixed] = lower[fixed]
    # fixed points leave the system: their terms move to the right-hand side
    g[1:] += off1 * x_fix[:-1]
    g[:-1] += off1 * x_fix[1:]
    g[2:] += off2 * x_fix[:-2]
    g[:-2] += off2 * x_fix[2:]
    d1[fixed[:-1] | fixed[1:]] = 0.0
    d2[fixed[:-2] | fixed[2:]] = 0.0
    d0[fixed] = 1.0
    g[fixed] = -x_fix[fixed]
    free = ~fixed
    has_l, has_u = free & np.isfinite(lower), free & np.isfinite(upper)
    x = x_fix.copy()
    both = has_l & has_u
    x[both] = 0.5 * (lower[both] + upper[both])
    x[has_l & ~both] = np.maximum(x[has_l & ~both], lower[has_l & ~both] + 1e-2)
    x[has_u & ~both] = np.minimum(x[has_u & ~both], upper[has_u & ~both] - 1e-2)
    sl, su = np.ones(n), np.ones(n)
    sl[has_l] = np.maximum(x[has_l] - lower[has_l], 1e-2)
    su[has_u] = np.maximum(upper[has_u] - x[has_u], 1e-2)
    zl, zu = np.where(has_l, 1.0, 0.0), np.where(has_u, 1.0, 0.0)
    count = max(int(has_l.sum() + has_u.sum()), 1)
    ab = np.zeros((3, n))

    def multiply(v):
        out = d0 * v
        out[:-1] += d1 * v[1:]
        out[1:] += d1 * v[:-1]
        out[:-2] += d2 * v[2:]
        out[2:] += d2 * v[:-2]
        return out

    for _ in range(iterations):
        rd = multiply(x) + g - zl + zu
        rl = np.where(has_l, x - lower - sl, 0.0)
        ru = np.where(has_u, upper - x - su, 0.0)
        mu = (sl[has_l] @ zl[has_l] + su[has_u] @ zu[has_u]) / count
        if (mu < tol * 1e-2 and max(np.max(np.abs(rd[free]), initial=0.0), np.max(np.abs(rl)), np.max(np.abs(ru))) < tol):
            break
        ab[0] = d0 + np.where(has_l, zl / sl, 0.0) + np.where(has_u, zu / su, 0.0)
        ab[1, :-1] = d1
        ab[2, :-2] = d2

        def step(cl, cu):
            right = -rd + np.where(has_l, (cl - zl * rl) / sl, 0.0) - np.where(has_u, (cu - zu * ru) / su, 0.0)
            dx = solveh_banded(ab, right, lower=True, check_finite=False)
            dsl, dsu = np.where(has_l, dx + rl, 0.0), np.where(has_u, -dx + ru, 0.0)
            dzl = np.where(has_l, (cl - zl * dsl) / sl, 0.0)
            dzu = np.where(has_u, (cu - zu * dsu) / su, 0.0)
            return dx, dsl, dsu, dzl, dzu

        def longest(v, dv):
            neg = dv < 0
            return min(1.0, float(np.min(-v[neg] / dv[neg]))) if neg.any() else 1.0

        try:
            _, dsl, dsu, dzl, dzu = step(-sl * zl, -su * zu)
            ap = min(longest(sl, dsl), longest(su, dsu))
            ad = min(longest(zl, dzl), longest(zu, dzu))
            gap = ((sl[has_l] + ap * dsl[has_l]) @ (zl[has_l] + ad * dzl[has_l])
                   + (su[has_u] + ap * dsu[has_u]) @ (zu[has_u] + ad * dzu[has_u])) / count
            sigma = (gap / max(mu, 1e-300)) ** 3
            dx, dsl, dsu, dzl, dzu = step(sigma * mu - sl * zl - dsl * dzl, sigma * mu - su * zu - dsu * dzu)
        except (LinAlgError, ValueError):
            return None
        ap = 0.995 * min(longest(sl, dsl), longest(su, dsu))
        ad = 0.995 * min(longest(zl, dzl), longest(zu, dzu))
        x, sl, su, zl, zu = x + ap * dx, sl + ap * dsl, su + ap * dsu, zl + ad * dzl, zu + ad * dzu
    else:
        return None
    x[fixed] = x_fix[fixed]
    return np.clip(x, lower, upper)


def solve_offsets(ds, kappa, v, lower, upper, slope0=0.0, grip=1.0, length=1.0, a_ref=5.0):
    """The offsets at n points ds apart (arrays kappa, v of the raceline there), bounds lower/upper. lower[0] ==
    upper[0] is where the car is, slope0 its d'; the last two points are held at 0. None if there is no such line."""
    n = len(kappa)
    lower, upper = lower.copy(), upper.copy()
    lower[-2:] = 0.0
    upper[-2:] = 0.0
    start = 0.5 * (lower[0] + upper[0])
    lower[1] = upper[1] = start + ds * slope0
    if not (lower[1] <= upper[1]):
        return None
    w = grip * (v[1:-1] ** 2 / a_ref) ** 2 * ds       # (lateral acceleration per curvature) ^ 2
    q = length * ds / np.maximum(0.5 * (v[:-1] + v[1:]), 0.5)    # time per d'^2
    side, mid = 1.0 / ds ** 2, -2.0 / ds ** 2 + kappa[1:-1] ** 2
    diag, off1, off2 = np.zeros(n), np.zeros(n - 1), np.zeros(n - 2)
    diag[:-2] += w * side * side
    diag[1:-1] += w * mid * mid
    diag[2:] += w * side * side
    off1[:-1] += w * side * mid
    off1[1:] += w * mid * side
    off2 += w * side * side
    c = q / ds ** 2
    diag[:-1] += c
    diag[1:] += c
    off1 -= c
    return box_qp(diag + 1e-9, off1, off2, np.zeros(n), lower, upper)


def speed_profile(ds, kappa, v_max, a_lat, a_acc, a_brake, v_start=None):
    """Quasi-steady-state speeds: (m candidates x n points) of curvature kappa, ds (m x n - 1) between the points.
    Friction-ellipse limited forward and backward passes, the v_start at the first point (m) if given."""
    limit = np.minimum(np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-4)), v_max)
    v = limit.copy()
    n = v.shape[1]
    if v_start is not None:
        v[:, 0] = np.minimum(v[:, 0], v_start)

    def along(vi, i, step, a_long):
        lat = vi ** 2 * np.abs(kappa[:, i]) / a_lat
        return np.sqrt(vi ** 2 + 2 * a_long * np.sqrt(np.maximum(1.0 - lat ** 2, 0.0)) * step)

    for i in range(n - 1):
        v[:, i + 1] = np.minimum(v[:, i + 1], along(v[:, i], i, ds[:, i], a_acc))
    for i in range(n - 2, -1, -1):
        v[:, i] = np.minimum(v[:, i], along(v[:, i + 1], i + 1, ds[:, i], a_brake))
    return v


def path_curvature(paths, span=2):
    """Paths (m x n x 2): (ds m x n - 1, curvature m x n), by the circle through the points span either side."""
    delta = np.diff(paths, axis=1)
    ds = np.maximum(np.hypot(delta[..., 0], delta[..., 1]), 1e-6)
    heading = np.unwrap(np.arctan2(delta[..., 1], delta[..., 0]), axis=1)
    s = np.concatenate((np.zeros((len(paths), 1)), np.cumsum(ds, axis=1)), axis=1)
    n = paths.shape[1]
    kappa = np.zeros((len(paths), n))
    i = np.arange(n)
    a, b = np.maximum(i - span, 0), np.minimum(i + span, n - 1)
    head_a, head_b = heading[:, np.minimum(a, n - 2)], heading[:, np.minimum(b - 1, n - 2)]
    first, last = np.minimum(a, n - 2), np.minimum(b - 1, n - 2)
    apart = 0.5 * (s[:, last] + s[:, last + 1]) - 0.5 * (s[:, first] + s[:, first + 1])   # between the edges' middles
    kappa[:] = (head_b - head_a) / np.maximum(apart, 1e-6)
    return ds, kappa


def time_lost(paths, reference, v_start, v_max, a_lat, a_acc, a_brake, v_end=None):
    """Seconds each path (m x n x 2) takes more than reference (n x 2), between the same two points, on quasi-steady
    speed profiles with the same limits. Returns (lost m, times m, speeds m x n)."""
    both = np.concatenate((paths, reference[None]), axis=0)
    ds, kappa = path_curvature(both)
    v = speed_profile(ds, kappa, v_max, a_lat, a_acc, a_brake, v_start)
    t = np.sum(2 * ds / np.maximum(v[:, :-1] + v[:, 1:], 1e-3), axis=1)
    return t[:-1] - t[-1], t[:-1], v[:-1]

"""Geometry of overtaking segments, shared by tum_segment.py (TUM venv) and plan_segment.py (ROS Python): numpy only.

Two lines are involved. The reference line is the smoothed centerline TUM optimizes on (points every
stepsize_reg, offsets n positive to the left, heading error xi). The raceline is TUM's exported global line
(raceline.csv). A segment is an open stretch of the reference line from the car's point to a window end.

Window ends (docs: Overtaking System Outline), on the raceline, after s_from (where the pass is done, see
pass_point; the car without an opponent):
  A  exit of the first corner that starts after s_from
  B  braking point of the corner after A's corner; a straight shorter than min_straight_m (a chicane) is
     driven through, i.e. the braking point of the next corner is used instead
  C  as B, n_corners corners later
A corner is a run of raceline points where |kappa|, averaged over smooth_m, is >= kappa_min and which turns
by at least min_turn: the average joins a corner the raw curvature's noise cuts in two, the turn drops bumps.
Every window is capped (cap_window): a C window on a track with few corners can be longer than a lap.
"""
import numpy as np


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def tangents(normals_left):
    """Unit tangents (direction of travel) from left normals."""
    nl = np.asarray(normals_left)
    return np.column_stack((nl[:, 1], -nl[:, 0]))


def project_point(line_xy, s_line, pt, closed=True):
    """(s, d) of point pt on a polyline: s along the line, d positive to its left (nearest segment)."""
    a = np.asarray(line_xy, dtype=float)
    b = np.roll(a, -1, axis=0) if closed else a[1:]
    a = a if closed else a[:-1]
    ab = b - a
    seg_len2 = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12)
    u = np.clip(np.einsum('ij,ij->i', np.asarray(pt) - a, ab) / seg_len2, 0.0, 1.0)
    foot = a + u[:, None] * ab
    dist = np.hypot(*(np.asarray(pt) - foot).T)
    j = int(np.argmin(dist))
    seg_len = np.sqrt(seg_len2[j])
    d = np.cross(ab[j], np.asarray(pt) - a[j]) / seg_len
    return float(s_line[j] + u[j] * seg_len), float(d)


def cross_normals(ref_xy, ref_nl, line_xy, max_n):
    """Where each reference normal crosses a closed polyline.

    Returns (n, j): offset along the left normal (|n| <= max_n, the crossing nearest the reference point) and the
    fractional index into line_xy (segment j, j + 1). NaN where a normal does not cross the line.
    """
    a = np.asarray(line_xy, dtype=float)
    b = np.roll(a, -1, axis=0)
    ab = b - a
    n_out = np.full(len(ref_xy), np.nan)
    j_out = np.full(len(ref_xy), np.nan)
    for i, (p, nl) in enumerate(zip(np.asarray(ref_xy), np.asarray(ref_nl))):
        # p + n * nl = a + u * ab
        den = nl[0] * ab[:, 1] - nl[1] * ab[:, 0]
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        ap = a - p
        n = (ap[:, 0] * ab[:, 1] - ap[:, 1] * ab[:, 0]) / den
        u = (ap[:, 0] * nl[1] - ap[:, 1] * nl[0]) / den
        hit = ok & (u >= 0.0) & (u <= 1.0) & (np.abs(n) <= max_n)
        if hit.any():
            k = np.flatnonzero(hit)[np.argmin(np.abs(n[hit]))]
            n_out[i] = n[k]
            j_out[i] = k + u[k]
    return n_out, j_out


def interp_closed(values, j_frac):
    """Linear interpolation of per-point values of a closed line at fractional indices."""
    v = np.asarray(values, dtype=float)
    j0 = np.floor(j_frac).astype(int) % len(v)
    j1 = (j0 + 1) % len(v)
    u = j_frac - np.floor(j_frac)
    return (1 - u) * v[j0] + u * v[j1]


def interp_closed_s(s_line, lap, j_frac):
    """Arc length at fractional indices of a closed line (the closing segment runs from the last point to lap)."""
    s = np.append(np.asarray(s_line, dtype=float), lap)
    j0 = np.floor(j_frac).astype(int) % len(s_line)
    u = j_frac - np.floor(j_frac)
    return s[j0] + u * (s[j0 + 1] - s[j0])


def unwrap_lap(s, lap):
    """Arc lengths along a closed line at successive points (each in [0, lap)), unwrapped past the lap end.
    Raises ValueError unless they then strictly increase within one lap (every step, the closing one from the
    last point back to the first included, goes forward): otherwise a reference normal reached another part of
    the line (e.g. across a thin wall at a hairpin), and interpolating between the two would be garbage."""
    s = np.asarray(s, dtype=float)
    ds = np.diff(s)
    ds[ds < -lap / 2] += lap
    s = np.concatenate(([s[0]], s[0] + np.cumsum(ds)))
    back = np.flatnonzero(ds <= 0)
    if len(back):
        i = int(back[0])
        raise ValueError('the raceline is crossed out of order: from reference point %d to %d its arc length goes '
                         'from %.2f to %.2f m (%d such steps); a reference normal reaches another part of the '
                         'raceline, e.g. across a thin wall at a hairpin' % (i, i + 1, s[i] % lap, s[i + 1] % lap,
                                                                            len(back)))
    if s[-1] - s[0] >= lap:
        raise ValueError('the raceline is crossed out of order: the reference points span %.2f m of a %.2f m lap; '
                         'a reference normal reaches another part of the raceline, e.g. across a thin wall at a '
                         'hairpin' % (s[-1] - s[0], lap))
    return s


def corners(kappa, kappa_min):
    """Corners of a closed line as (start, end) index pairs, end inclusive; a corner may wrap past the last point."""
    hot = np.abs(np.asarray(kappa)) >= kappa_min
    if hot.all():
        return [(0, len(hot) - 1)]
    if not hot.any():
        return []
    # start the scan on a straight point so no corner is cut in two by the array ends
    off = int(np.flatnonzero(~hot)[0])
    h = np.roll(hot, -off)
    out, i, n = [], 0, len(h)
    while i < n:
        if h[i]:
            j = i
            while j + 1 < n and h[j + 1]:
                j += 1
            out.append(((i + off) % n, (j + off) % n))
            i = j + 1
        else:
            i += 1
    return sorted(out)


def track_corners(kappa, ds, kappa_min, smooth_m=1.0, min_turn=np.radians(30.0)):
    """Corners of a closed line (see the module doc) as (start, end) index pairs; ds: length of each point's
    segment to the next. The average is of signed kappa, so the two turns of a chicane stay apart."""
    kappa, ds = np.asarray(kappa, dtype=float), np.asarray(ds, dtype=float)
    k = kappa
    half = int(round(smooth_m / 2 / max(float(np.median(ds)), 1e-6)))
    if half > 0:
        k = np.convolve(np.concatenate((kappa[-half:], kappa, kappa[:half])), np.ones(2 * half + 1) / (2 * half + 1),
                        'valid')
    out = []
    for i, j in corners(k, kappa_min):
        run = np.arange(i, j + 1 + (len(k) if j < i else 0)) % len(k)
        if abs(np.sum(kappa[run] * ds[run])) >= min_turn:
            out.append((i, j))
    return out


def braking_point(ax, corner_start):
    """First point of the deceleration run that ends at corner_start (the corner start itself if none)."""
    n = len(ax)
    i = corner_start
    for _ in range(n):
        prev = (i - 1) % n
        if ax[prev] >= 0.0:
            break
        i = prev
    return i


def window_end(raceline, s_from, mode, kappa_min=0.2, min_straight_m=1.0, n_corners=3, smooth_m=1.0,
               min_turn=np.radians(30.0)):
    """Raceline arc length (unwrapped, > s_from) where a segment of the given mode ends; see the module doc.

    raceline: (open rows [s, x, y, psi, kappa, vx, ax], lap length), as raceline_lap() returns it;
    s_from may be any real number (unwrapped arc length).
    """
    if mode not in ('A', 'B', 'C'):
        raise ValueError('mode must be A, B or C, not %r' % mode)
    rl, lap = raceline
    s, kappa, ax = rl[:, 0], rl[:, 4], rl[:, 6]
    cs = track_corners(kappa, np.diff(np.append(s, lap)), kappa_min, smooth_m, min_turn)
    if not cs:
        raise ValueError('no corners on the raceline (kappa_min = %.2f 1/m too high?)' % kappa_min)

    # each corner as unwrapped (braking point, start, exit) on the first lap, then repeated on later laps
    one_lap = []
    for i, j in cs:
        b = braking_point(ax, i)
        start = s[i]
        brake = s[b] - (lap if b > i else 0.0)
        exit_ = s[j] + (lap if j < i else 0.0)
        one_lap.append((brake, start, exit_))
    base = np.floor(s_from / lap) * lap
    laps_needed = 2 + (n_corners if mode == 'C' else 1) // len(one_lap) + 1
    unrolled = sorted((b + base + k * lap, st + base + k * lap, e + base + k * lap)
                      for k in range(-1, laps_needed + 1) for b, st, e in one_lap)
    ahead = [c for c in unrolled if c[1] > s_from]

    if mode == 'A':
        return ahead[0][2]
    count = 1 if mode == 'B' else n_corners
    last_exit = ahead[0][2]
    for brake, _, exit_ in ahead[1:]:
        if brake - last_exit >= min_straight_m:
            count -= 1
            if count == 0:
                return brake
        last_exit = exit_            # a chicane (straight too short) is driven through
    raise ValueError('window end not found')


def cap_window(s_start, s_end, lap, max_length_m, max_lap_fraction):
    """(s_end, capped): the window [s_start, s_end] cut to max_length_m and to max_lap_fraction of a lap."""
    longest = min(max_length_m, max_lap_fraction * lap)
    if s_end - s_start > longest:
        return s_start + longest, True
    return s_end, False


def pass_point(s_nodes, t_nodes, s_opp_at, clear_s):
    """First point where we are clear_s (centre to centre) ahead of the opponent, None if we never are.

    s_nodes, t_nodes: our unwrapped arc length and time at each point, s_opp_at(t): the opponent's predicted
    unwrapped arc length on the same line, in the same metres.
    """
    t = np.asarray(t_nodes, dtype=float)
    ahead = np.flatnonzero(np.asarray(s_nodes) - s_opp_at(t) >= clear_s)
    return int(ahead[0]) if len(ahead) else None


def raceline_point(raceline, s, d=0.0):
    """(x, y) at raceline arc length(s) s (any lap), d to its left; raceline as raceline_lap() returns it."""
    rl, lap = raceline
    s = np.asarray(s, dtype=float) % lap
    sp = np.append(rl[:, 0], lap)

    def at(v):
        return np.interp(s, sp, np.append(v, v[0]))
    yaw = np.arctan2(at(np.sin(rl[:, 3])), at(np.cos(rl[:, 3]))) + np.pi / 2   # TUM psi: 0 = north
    return at(rl[:, 1]) - d * np.sin(yaw), at(rl[:, 2]) + d * np.cos(yaw)


def raceline_lap(traj_closed):
    """Split TUM's closed export (last row repeats the first at s = lap) into (open rows, lap length)."""
    t = np.asarray(traj_closed, dtype=float)
    if len(t) > 1 and np.allclose(t[0, 1:3], t[-1, 1:3]):
        return t[:-1], float(t[-1, 0])
    seg = np.hypot(*(t[0, 1:3] - t[-1, 1:3]))
    return t, float(t[-1, 0] + seg)


def block_widths(w_right, w_left, mask, n_opp, half_width, side):
    """Track widths with the band [n_opp - half_width, n_opp + half_width] removed where mask is set.

    side 'left': we pass on its left, so the right bound moves up to n_opp + half_width;
    side 'right': the left bound moves down to n_opp - half_width. Widths only ever shrink.
    The band is in bound terms: TUM keeps our centre width_opt / 2 inside it.
    """
    w_r, w_l = np.array(w_right, dtype=float), np.array(w_left, dtype=float)
    n_opp = np.broadcast_to(np.asarray(n_opp, dtype=float), w_r.shape)
    if side == 'left':
        w_r = np.where(mask, np.minimum(w_r, -(n_opp + half_width)), w_r)
    elif side == 'right':
        w_l = np.where(mask, np.minimum(w_l, n_opp - half_width), w_l)
    else:
        raise ValueError("side must be 'left' or 'right'")
    return w_r, w_l


def alongside(s_nodes, t_nodes, s_opp_at, gap_s, m_s0, k_s):
    """Points where we would be next to the opponent: |s_i - s_opp(t_i)| <= gap_s + m_s0 + k_s * t_i.

    s_nodes: our unwrapped arc length at each point, t_nodes: our time there, s_opp_at(t): the opponent's
    predicted unwrapped arc length at time t, on the same line and in the same metres as s_nodes (tum_segment:
    the raceline's, where car lengths are real distances). gap_s = half of both car lengths.
    """
    t = np.asarray(t_nodes, dtype=float)
    return np.abs(np.asarray(s_nodes) - s_opp_at(t)) <= gap_s + m_s0 + k_s * t


def segment_time(h, n, xi, v, kappa, beta=0.0):
    """Time at every point of a segment from its states: dt/ds = (1 - n kappa) / (v cos(xi + beta))."""
    sf = (1.0 - np.asarray(n) * np.asarray(kappa)) / (np.maximum(np.asarray(v), 0.1)
                                                      * np.cos(np.asarray(xi) + beta))
    return np.concatenate(([0.0], np.cumsum(0.5 * (sf[:-1] + sf[1:]) * np.asarray(h))))

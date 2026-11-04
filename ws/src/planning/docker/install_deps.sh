#!/usr/bin/env bash
# Planning dependencies, run once at image build (see the root Dockerfile).
#  1. apt: Python libs the planning package uses from ROS's Python (centerline extraction, preview).
#  2. TUM global_racetrajectory_optimization (mintime) in its own venv, /opt/tum_venv, so its
#     pinned numpy/casadi/scipy never touch ROS's Python. build_raceline calls it via subprocess.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TUM_REPO=https://github.com/TUMFTM/global_racetrajectory_optimization.git
TUM_COMMIT=a9995e2f5407f22eb7fb9dceac2b71a35276bb41   # master, 2021-04-01 (latest upstream)
TUM_DIR=/opt/tum_optimizer
VENV=/opt/tum_venv

apt-get update
apt-get install -y --no-install-recommends \
    python3-scipy python3-skimage python3-matplotlib python3-yaml python3-pytest \
    python3-venv python3-dev build-essential git
rm -rf /var/lib/apt/lists/*

git clone --quiet "$TUM_REPO" "$TUM_DIR"
git -C "$TUM_DIR" checkout --quiet "$TUM_COMMIT"
# Scaling factors / yaw-rate bound configurable for a 1/10 car, the yaw-moment arm fix, and open
# segments (pars optim_opts segment: fixed start state, terminal cost, no lap closure) for planning/tum_segment.py.
git -C "$TUM_DIR" apply "$HERE/tum_mintime.patch"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --no-cache-dir --upgrade pip
"$VENV/bin/pip" install --no-cache-dir -r "$HERE/tum_requirements.txt"
# tph pins quadprog==0.1.7 (no py3.10 build); quadprog comes from tum_requirements.txt instead.
"$VENV/bin/pip" install --no-cache-dir --no-deps trajectory-planning-helpers==0.79
# tph 0.79 vs scipy 1.11: one-line fix in spline_approximation.dist_to_p
patch -p1 -d "$VENV/lib/python3.10/site-packages" < "$HERE/tph_scipy.patch"

# Smoke test: imports and a trivial IPOPT solve (casadi wheels bundle IPOPT + MUMPS).
cd "$TUM_DIR"
MPLBACKEND=Agg "$VENV/bin/python" -c "
import casadi as ca, quadprog, sklearn, trajectory_planning_helpers, opt_mintime_traj, helper_funcs_glob
x = ca.SX.sym('x')
s = ca.nlpsol('s', 'ipopt', {'x': x, 'f': (x - 1) ** 2}, {'ipopt.print_level': 0, 'print_time': 0})
assert abs(float(s(x0=0)['x']) - 1) < 1e-6
print('TUM mintime venv OK, casadi', ca.__version__)
"

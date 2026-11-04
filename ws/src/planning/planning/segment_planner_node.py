#!/usr/bin/env python3
"""Minimum-time overtaking segments in the race: tum_segment.py around what the lidar sees, driven by pure pursuit.

race.launch.py segments:=true runs this instead of obstacle_spliner_node. It uses that node's scan placement and
clusters, and publishes the same /local_path + /local_path/valid. An obstacle within block_distance of the raceline
is planned around: all three windows (segment_geometry: first corner, second corner, after the second corner), both
sides, by SOLVERS long-lived solver processes (tum_segment.py --serve), one side per solve, so both sides of the first
window start at once, each solve at most MAX_SOLVE s. The first viable segment is driven at once, so an impossible side costs
nothing; one losing less time replaces it only while the car has not reached it. A plan stays for that obstacle, also
on later laps, while it clears what the scan shows, starting from the car wherever that is.

The solver warms up from the start, in idle time: its first answer would otherwise take seconds (the process, the
track, a solver per window length). Until there is a plan the car stays on the raceline; closer than the stopping distance it stops and waits. No viable
segment around an obstacle really blocking the raceline: it stays stopped.

Limits of this first version: obstacles are taken as standing still (v 0), and one group per plan: a box further along
the window is not known to the solver until it is seen, and then the plan is made again. Only the segment's path is
used: pure pursuit keeps the raceline's speeds.

Publishes /segments/lines (std_msgs/String, JSON: {"status": text, "lines": [{"state": "chosen" | "viable" |
"refused", "name": "A_left", "xy": [[x, y], ...]}]}) for autodrive/map_view_node.py.
"""
import json
import math
import os
import queue
import re
import subprocess
import tempfile
import threading

import numpy as np
import rclpy
from std_msgs.msg import String

from planning.build_raceline import default_config
from planning.obstacle_spliner_node import CAR_EXEMPT, FROM_CAR_BLEND, HOLD_OBSTACLE, HOLD_WALL, ObstacleSpliner, seconds
from planning.raceline import TUM_PYTHON, read_raceline

DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tum_segment.py')
MODES = ('A', 'B', 'C')
MIN_BOX = 0.3      # m: along the line the lidar sees only a near face
WARM_EVERY = 3.0   # m: the solver is warmed up from starts this far apart along the raceline, in idle time
SOLVERS = 2
MAX_SOLVE = 1.5    # s: an impossible side can take many seconds to be proved so
SAME_S, SAME_D = 0.6, 0.35   # m: an obstacle this close to a plan's is the same one
RESTART_AFTER = 300   # solves: the solver process keeps a solver per window length, start it afresh now and then


def plan_offsets(project, s_rail, lap, rows):
    """A segment's rows as offsets from the raceline at each raceline point (0 outside the segment)."""
    xy = np.asarray(rows)[:, 1:3]
    seg = np.hypot(*np.diff(xy, axis=0).T)
    dense = np.vstack([np.linspace(a, b, max(int(d / 0.05), 1), endpoint=False) for a, b, d in zip(xy[:-1], xy[1:], seg)]
                      + [xy[-1:]])
    s, d, _ = project(dense)
    s = np.unwrap(s, period=lap)
    if s[-1] < s[0]:
        raise ValueError('segment runs against the raceline')
    k = np.concatenate(([0], 1 + np.flatnonzero(np.diff(s) > 1e-6)))
    s, d = s[k], d[k]
    rel = (s_rail - s[0]) % lap
    out = np.interp(rel, s - s[0], d, right=0.0)
    out[rel > s[-1] - s[0]] = 0.0
    return out


class Solver:
    """tum_segment.py --serve in the TUM venv: one request at a time, on a thread."""

    def __init__(self, log):
        self.log, self.proc, self.count = log, None, 0
        self.jobs, self.done = queue.Queue(), queue.Queue()
        self.busy = False
        threading.Thread(target=self.run, daemon=True).start()

    def start(self):
        err = open('/tmp/segment_solver.log', 'a')
        self.proc = subprocess.Popen([TUM_PYTHON, '-I', DRIVER, '--serve'], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=err, text=True)
        self.count = 0

    def run(self):
        while True:
            tag, req = self.jobs.get()
            try:
                if self.proc is None or self.proc.poll() is not None or self.count >= RESTART_AFTER:
                    if self.proc is not None:
                        self.proc.kill()
                    self.start()
                self.proc.stdin.write(json.dumps(req) + '\n')
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                self.count += 1
                resp = json.loads(line) if line else {'error': 'the solver process ended'}
            except (OSError, ValueError) as e:
                resp = {'error': str(e)}
            self.done.put((tag, resp))

    def submit(self, tag, req):
        self.busy = True
        self.jobs.put((tag, req))

    def close(self):
        if self.proc is not None:
            self.proc.kill()


class SegmentPlanner(ObstacleSpliner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        p = self.declare_parameter
        self.raceline_csv = self.get_parameter('raceline_path').value
        self.centerline = p('centerline_path', os.path.join(os.path.dirname(self.raceline_csv), 'centerline.csv')).value
        self.base_v = read_raceline(self.raceline_csv)[:self.n, 5]   # planned speed at each raceline point
        self.ego_rel = None
        self.block_distance = float(p('block_distance', 0.205).value)   # m from the raceline: it blocks the line
        self.lead = float(p('lead', 3.0).value)                # m before the obstacle a segment starts on the raceline
        self.max_decel = float(p('stop_decel', 2.5).value)     # m/s^2, for the stopping distance
        self.stop_margin = float(p('stop_margin', 1.0).value)  # m from the lidar's near face, stopped
        self.speed_scale = float(p('speed_scale', 0.5).value)
        step = float(p('stepsize', 0.4).value)                 # m between the solver's points (lap: 0.2)
        with open(p('config', default_config()).value) as f:
            ini = f.read()
        if '"stepsize_reg": 0.2' not in ini:
            raise ValueError('config: no "stepsize_reg": 0.2 to replace')
        fd, self.ini = tempfile.mkstemp(prefix='segment_', suffix='.ini')
        with os.fdopen(fd, 'w') as f:
            f.write(re.sub(r'"stepsize_reg": 0\.2', f'"stepsize_reg": {step}', ini))
        self.lines_pub = self.create_publisher(String, '/segments/lines', 10)
        self.solvers = [Solver(self.get_logger()) for _ in range(SOLVERS)]
        self.plans, self.next_id = [], 0
        self.car_v, self.status = 0.0, 'no obstacle'
        self.last_pose = None
        self.warm = [[float(x) for x in np.arange(0.0, self.lap, WARM_EVERY)] for _ in range(SOLVERS)]
        self.create_timer(0.5, self.send_lines)

    # ---------------------------------------------------------------- the plans

    def room_order(self, s_mid, d_right, d_left):
        """The sides, the one with more track beside the obstacle first."""
        i = int(np.argmin(np.abs(self.s - s_mid % self.lap)))

        def room(edge, sign):
            ds = np.arange(0.0, 3.0, 0.05)
            free = self.clearances(self.base_xy[i] + (edge + sign * ds)[:, None] * self.normal[i]) > 0.0
            return float(ds[np.argmin(free)]) if not free.all() else 3.0
        return ['left', 'right'] if room(d_left, 1.0) >= room(d_right, -1.0) else ['right', 'left']

    def request(self, plan, job):
        mode, side = job
        start = plan['start']
        return {'centerline': self.centerline, 'raceline': self.raceline_csv, 'ini': self.ini, 'start': start,
                'opponent': {'s': plan['s'] % self.lap, 'gap': plan['gap'], 'd': plan['d'], 'v': 0.0,
                             'length': plan['length'], 'width': plan['width']},
                'window': {'mode': mode}, 'sides': [side], 'margins': {'m_s0': 0.3, 'k_s': 0.0, 'm_d': 0.15},
                'ipopt': {'ipopt.max_cpu_time': MAX_SOLVE}}

    def warm_up(self):
        """One more window length built, on a solver nothing else needs."""
        for solver, todo in zip(self.solvers, self.warm):
            if solver.busy or not todo:
                continue
            i = int(np.argmin(np.abs(self.s - todo.pop(0))))
            yaw = math.atan2(self.tan[i, 1], self.tan[i, 0])
            solver.submit(('warm', 'A'), {
                'centerline': self.centerline, 'raceline': self.raceline_csv, 'ini': self.ini, 'opponent': None,
                'start': {'x': float(self.base_xy[i, 0]), 'y': float(self.base_xy[i, 1]), 'yaw': yaw,
                          'v': float(self.base_v[i])}, 'window': {'mode': 'A'}})

    def new_plan(self, ego_s, near, far, d_right, d_left, pose):
        s_mid, length = ego_s + (near + far) / 2, max(far - near + MIN_BOX, MIN_BOX)
        plan = {'id': self.next_id, 's': s_mid, 'd': (d_right + d_left) / 2, 'length': length,
                'width': max(d_left - d_right, 0.05), 'jobs': None, 'lines': [], 'chosen': None, 'offset': None,
                'time': 0.0}
        self.next_id += 1
        first, second = self.room_order(s_mid, d_right, d_left)
        plan['jobs'] = [(m, side) for m in MODES for side in (first, second)]
        if near > self.lead + 1.0:     # on the raceline before it: the same plan works on every lap
            s0 = ego_s + near - self.lead
            i = int(np.argmin(np.abs(self.s - s0 % self.lap)))
            yaw = math.atan2(self.tan[i, 1], self.tan[i, 0])
            plan['start'] = {'x': float(self.base_xy[i, 0]), 'y': float(self.base_xy[i, 1]), 'yaw': yaw,
                             'v': float(self.base_v[i])}
            plan['gap'] = s_mid - s0
        else:                          # too close for that: from the car as it is
            x, y, yaw = pose
            plan['start'] = {'x': x, 'y': y, 'yaw': yaw, 'v': max(self.car_v / self.speed_scale, 0.5)}
            plan['gap'] = s_mid - ego_s
        self.plans.append(plan)
        self.get_logger().info(f"plan {plan['id']}: obstacle at s {s_mid % self.lap:.1f}, d {plan['d']:+.2f}, "
                               f"{plan['width']:.2f} m wide, {near:.1f} m ahead")
        return plan

    def take_results(self):
        for solver in self.solvers:
            while not solver.done.empty():
                (pid, mode), resp = solver.done.get()
                solver.busy = False
                plan = next((q for q in self.plans if q['id'] == pid), None)
                if plan is None:
                    if pid == 'warm' and not any(self.warm) and not any(x.busy for x in self.solvers):
                        self.get_logger().info('solver warmed up')
                    continue
                if 'error' in resp:
                    self.get_logger().error(f"plan {pid} {mode}: {resp['error'][:300]}")
                    continue
                self.take_response(plan, pid, mode, resp)

    def take_response(self, plan, pid, mode, resp):
        lost_seed = resp['seed_time']
        for side, r in resp['sides'].items():
            plan['time'] += r['solve_time']
            name, why = f'{mode}_{side}', '; '.join(r['why'])
            if 'rows' not in r:
                self.get_logger().info(f'plan {pid} {name}: {why}')
                continue
            xy = np.asarray(r['rows'])[:, 1:3].round(3).tolist()
            lost = r['segment_time'] - lost_seed
            line = {'name': name, 'xy': xy, 'viable': bool(r['feasible']), 'lost': lost, 'rows': r['rows']}
            plan['lines'].append(line)
            self.get_logger().info(f"plan {pid} {name}: {'viable' if r['feasible'] else 'refused: ' + why}, "
                                   f"{lost:+.3f} s to the raceline, solved in {r['solve_time']:.2f} s")
            if r['feasible']:
                self.consider(plan, line)

    def consider(self, plan, line):
        """The viable line losing the least time; a better one only while the car has not reached the shift."""
        if plan['chosen'] is not None and line['lost'] >= plan['chosen']['lost']:
            return
        try:
            offset = plan_offsets(self.project, self.s, self.lap, line['rows'])
        except ValueError as e:
            self.get_logger().warning(f"plan {plan['id']} {line['name']}: {e}")
            return
        if plan['offset'] is not None and self.ego_rel is not None:
            shifted = np.abs(plan['offset']) > 0.02
            if shifted.any() and self.ego_rel[shifted].min() < 1.0 and (self.ego_rel[shifted] < 3.0).any():
                return     # the car is at it
        plan['chosen'], plan['offset'] = line, offset
        self.get_logger().info(f"plan {plan['id']}: driving {line['name']}")

    def send_lines(self):
        lines = []
        for plan in self.plans:
            for line in plan['lines']:
                state = 'chosen' if line is plan['chosen'] else 'viable' if line['viable'] else 'refused'
                lines.append({'state': state, 'name': line['name'], 'xy': line['xy']})
        self.lines_pub.publish(String(data=json.dumps({'status': self.status, 'lines': lines})))

    # ---------------------------------------------------------------- each tick

    def tick(self):
        now = self.now()
        self.take_results()
        self.warm_up()
        if (self.pose is None or self.scan is None or now - self.pose_time > self.max_pose_age
                or now - self.scan_time > self.max_scan_age):
            return self.publish(self.base_xy, False)
        newest, pose, scan = seconds(self.scan.header.stamp), None, None
        for scan in reversed(self.scans):
            if newest - seconds(scan.header.stamp) > self.max_pose_wait:
                break
            pose = self.pose_at(seconds(scan.header.stamp))
            if pose is not None:
                break
        if pose is None:
            return None    # the pose stream will catch up: the last path stands
        position, q = self.pose.pose.position, self.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        here = (float(position.x), float(position.y), self.pose_time)
        if self.last_pose is not None and here[2] > self.last_pose[2]:
            self.car_v = float(np.hypot(here[0] - self.last_pose[0], here[1] - self.last_pose[1])
                               / (here[2] - self.last_pose[2]))
        self.last_pose = here
        ego_s = self.project([position.x, position.y])[0][0]
        self.ego_rel = (self.s - ego_s) % self.lap
        rel = (self.s - ego_s + self.lap / 2) % self.lap - self.lap / 2
        clusters = self.scan_clusters(scan, pose)
        blocking = []
        for xy in clusters:
            s_pts, d_pts, dist = self.project(xy)
            ahead = (s_pts - ego_s + self.lap / 2) % self.lap - self.lap / 2
            if (dist < self.block_distance).any() and ahead.min() < self.lookahead \
                    and ahead.max() + self.obstacle_length + self.half_width > 0:
                blocking.append((ahead.min(), ahead.max(), d_pts.min(), d_pts.max()))
        seen = np.vstack(clusters) if clusters else np.empty((0, 2))
        if not blocking:
            self.status = 'no obstacle on the raceline'
            # a plan still being driven, past its obstacle: keep to it until it has joined the raceline
            for plan in self.plans:
                if plan['offset'] is not None and self.on_shift(plan, rel):
                    return self.publish(self.path_of(plan), True)
            return self.publish(self.base_xy, True)
        blocking.sort()
        near, far, d_right, d_left = blocking[0]
        for k in blocking[1:]:
            if k[0] <= far + self.obstacle_length:
                far, d_right, d_left = max(far, k[1]), min(d_right, k[2]), max(d_left, k[3])
        s_mid, d_mid = (ego_s + (near + far) / 2) % self.lap, (d_right + d_left) / 2
        plan = None
        for q_ in self.plans:
            ds = abs((q_['s'] - s_mid + self.lap / 2) % self.lap - self.lap / 2)
            if ds < SAME_S and abs(q_['d'] - d_mid) < SAME_D:
                plan = q_
                break
        if plan is None:
            self.plans = [q_ for q_ in self.plans if self.on_shift(q_, rel)]    # one obstacle's lines at a time
            plan = self.new_plan(ego_s, near, far, d_right, d_left, (float(position.x), float(position.y), yaw))
        # the path it would drive must still clear what the scan shows, and start where the car is
        if plan['offset'] is not None:
            path, moved = self.from_car(plan, rel, position, near)
            moved = moved & (rel >= CAR_EXEMPT)
            if self.margin_of(path, moved, seen, HOLD_WALL, HOLD_OBSTACLE) < 0.0:
                self.get_logger().warning(f"plan {plan['id']}: {plan['chosen']['name']} no longer clears the scan: "
                                          'planning again')
                self.plans.remove(plan)
                plan = self.new_plan(ego_s, near, far, d_right, d_left, (float(position.x), float(position.y), yaw))
        for solver in self.solvers:
            if not solver.busy and plan['jobs']:
                job = plan['jobs'].pop(0)
                solver.submit((plan['id'], job[0]), self.request(plan, job))
        solving = any(x.busy for x in self.solvers) or bool(plan['jobs'])
        if plan['offset'] is not None:
            self.status = (f"driving {plan['chosen']['name']} ({plan['chosen']['lost']:+.3f} s), "
                           f"{sum(x['viable'] for x in plan['lines'])} viable"
                           + (', still solving' if solving else f", solved in {plan['time']:.1f} s"))
            return self.publish(self.from_car(plan, rel, position, near)[0], True)
        stop_at = self.stop_margin + self.car_v ** 2 / (2 * self.max_decel)
        if near > stop_at:
            self.status = f'solving, {near:.1f} m to the obstacle'
            return self.publish(self.base_xy, True)
        self.status = 'stopped: solving' if solving else 'stopped: no viable segment around the obstacle'
        self.get_logger().warning(self.status, throttle_duration_sec=2.0)
        return self.publish(self.base_xy, False)

    def on_shift(self, plan, rel):
        """The car is on the plan's shift, or about to be."""
        if plan['offset'] is None:
            return False
        shifted = np.abs(plan['offset']) > 0.02
        return shifted.any() and (np.abs(rel[shifted]) < 1.0).any()

    def from_car(self, plan, rel, position, near):
        """The plan's path starting at the car: (path, the points off the raceline)."""
        ego_d = float(self.project([position.x, position.y])[1][0])
        offset = self.shift_from_car(rel, plan['offset'], ego_d, min(FROM_CAR_BLEND, max(near, 0.5)))
        return self.base_xy + offset[:, None] * self.normal, np.abs(offset) > 1e-9

    def path_of(self, plan):
        return self.base_xy + plan['offset'][:, None] * self.normal


def main():
    rclpy.init()
    node = SegmentPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for solver in node.solvers:
            solver.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

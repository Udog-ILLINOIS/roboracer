#!/usr/bin/env python3
"""Keyboard teleop: publishes /drive (AckermannDriveStamped) at a steady rate.

Speed is in m/s and goes through the adapter's speed loop (the VESC on the car), steering in rad.
Ported from AutoDRIVE's devkit teleop, which wrote raw throttle/steering commands instead.

Keys are smoothed the way racing games smooth digital input (Unity's input axes: a held key moves
the value at a set rate, releasing it lets the value return):
  - holding W/S ramps the speed at max_acceleration / brake_deceleration; releasing coasts down at
    coast_deceleration (engine braking, gentler than S)
  - holding A/D builds the requested turn toward full lock at steer_rate; releasing returns it at
    center_rate
  - the wheels only turn as far as the grip allows: the request is clipped so the lateral acceleration
    stays under max_lateral_accel, tan(steer) <= wheelbase * max_lateral_accel / v^2. Past the tyres'
    grip the front slides (understeer)
    and the car turns less, not more: measured in the sim, the yaw rate follows v * tan(steer) / wheelbase
    up to ~3.5-4 m/s^2 lateral, then saturates (2 m/s at full lock: 1.6 rad/s, at 0.35 rad: 2.0 rad/s)
  - traction control: while cornering, acceleration is cut to what the grip has left,
    ax <= sqrt(max_total_accel^2 - ay^2) (friction circle)
  - brake for grip: while the request is above what the grip allows (S not held), the car brakes at
    corner_deceleration until it is not, the wheels following the request as the speed drops. A short
    press is a gentle turn with little braking, a long hold a tight turn. It overrides W.
  - stability control: the gyro yaw rate (imu_topic) is compared with what the steering asks for,
    v * tan(steer) / wheelbase with v from odom_topic. Under slip_ratio of it for slip_confirm s means the
    front is sliding: the car brakes at slip_deceleration whatever the keys say, and the steering stops
    growing, until the yaw rate catches up.
A terminal only sends key presses (no releases), so a key counts as held while its auto-repeat keeps
arriving: key_repeat_delay after the first press, key_repeat_window after each repeat.

  ros2 run autodrive_bridge teleop_keyboard    (needs a terminal; keys are read from stdin)
"""
import math
import os
import re
import select
import signal
import sys
import termios
import time
import tty

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.signals import SignalHandlerOptions
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

HELP = """
---------------------------------------
RoboRacer keyboard teleop -> /drive
---------------------------------------
            W   E
        A   S   D
            X       R

W (hold) : accelerate {accel:.1f} m/s^2 to {max_speed:.1f} m/s; release to coast down {coast:.1f} m/s^2
S (hold) : brake {brake:.1f} m/s^2, then reverse to {max_reverse:.1f} m/s
A/D (hold): ask for a left/right turn, more the longer you hold; the wheels re-centre on release.
            The wheels only turn as far as the grip allows (lateral accel <= {max_lat:.1f} m/s^2,
            lock {max_steer:.3f} rad); asking for more brakes {corner:.1f} m/s^2 until it has grip (GRIP)
Auto brakes {slip:.1f} m/s^2 when the gyro shows the front sliding (status line: SLIP)
E   : brake to a stop
X   : stop now and zero steering
R   : stop and soft-reset the simulator
Ctrl+C : quit (sends a zero command)
---------------------------------------
"""


OPPOSITE = {'w': 's', 's': 'w', 'a': 'd', 'd': 'a'}
# arrow and other cursor keys: ESC [ params final byte (ESC [ A, ESC [ 1 ; 5 D for Ctrl+Left; lower-cased the
# A or D would steer) or ESC O + one byte
ESCAPE = re.compile(r'\x1b(?:\[[0-9;?]*[ -/]*[@-~]|O.)', re.S)
PARTIAL = re.compile(r'\x1b(?:\[[0-9;?]*[ -/]*|O)?$')   # an escape cut off at the end of a read


def split_escape(text):
    """(text, rest): rest is an escape sequence cut off at the end of the read, kept for the next one."""
    m = PARTIAL.search(text)
    return (text[:m.start()], text[m.start():]) if m else (text, '')


def handle_keys(text, now, held_until, car, repeat_delay, repeat_window):
    """Apply the keys read from the terminal (held_until: key -> time its hold expires). A key drops its
    opposite, so S after a W tap brakes and D after A steers right instead of cancelling out.
    Returns True if R asked for a simulator reset."""
    reset = False
    for key in ESCAPE.sub('', text).lower():
        if key in 'wsad':
            repeat = held_until.get(key, 0.0) >= now
            held_until[key] = now + (repeat_window if repeat else repeat_delay)
            held_until.pop(OPPOSITE[key], None)
        elif key == 'e':
            car.stopping = True
            held_until.pop('w', None)
            held_until.pop('s', None)
        elif key in ('x', 'r'):
            car.stop()
            held_until.clear()
            reset = reset or key == 'r'
    return reset


def approach(x, target, step):
    """Move x toward target by at most step."""
    return min(x + step, target) if x < target else max(x - step, target)


class Smoother:
    """Turns held keys into a speed and steering command with game-style rate limits."""

    def __init__(self, max_speed, max_reverse, accel, brake, coast, max_steer, steer_rate, center_rate,
                 wheelbase, max_lat_accel, max_total_accel, corner_decel,
                 slip_decel=4.0, slip_ratio=0.85, slip_min_yaw=0.3, slip_confirm=0.15):
        self.max_speed, self.max_reverse = max_speed, max_reverse
        self.accel, self.brake, self.coast = accel, brake, coast
        self.max_steer, self.steer_rate, self.center_rate = max_steer, steer_rate, center_rate
        self.wheelbase, self.max_lat_accel = wheelbase, max_lat_accel
        self.max_total_accel, self.corner_decel = max_total_accel, corner_decel
        self.slip_decel, self.slip_ratio = slip_decel, slip_ratio
        self.slip_min_yaw, self.slip_confirm = slip_min_yaw, slip_confirm
        self.speed = self.steer = 0.0
        self.request = 0.0      # rad, the turn A/D ask for; steer is it clipped to what the grip allows
        self.stopping = False   # E: brake to 0 until a throttle key is pressed
        self.slip_time = 0.0    # s the yaw rate has lagged the steering
        self.slipping = self.cornering = False

    def steer_limit(self):
        v2 = self.speed * self.speed
        if v2 < 1e-6:
            return self.max_steer
        return min(self.max_steer, math.atan(self.wheelbase * self.max_lat_accel / v2))

    def lat_accel(self):
        return self.speed * self.speed * math.tan(abs(self.steer)) / self.wheelbase

    def grip_left(self):
        """Longitudinal acceleration the friction circle leaves beside the current lateral one."""
        return math.sqrt(max(0.0, self.max_total_accel ** 2 - self.lat_accel() ** 2))

    def check_slip(self, dt, yaw_rate, v):
        """Understeer: the measured yaw rate stays under slip_ratio of what the steering asks for."""
        expected = v * math.tan(self.steer) / self.wheelbase
        if yaw_rate is None or v <= 0 or abs(expected) < self.slip_min_yaw:
            self.slip_time = 0.0
        elif yaw_rate * math.copysign(1.0, expected) < self.slip_ratio * abs(expected):
            self.slip_time += dt
        else:
            self.slip_time = 0.0
        self.slipping = self.slip_time >= self.slip_confirm

    def stop(self):
        self.speed = self.steer = self.request = 0.0
        self.stopping = False

    def step(self, dt, throttle, steer, yaw_rate=None, v=None):
        """throttle, steer in {-1, 0, 1}: W/S and A/D held (A is +1, left). yaw_rate (rad/s, gyro) and
        v (m/s, odometry) are measurements for stability control; None turns it off for this step."""
        if throttle:
            self.stopping = False
        self.check_slip(dt, yaw_rate, self.speed if v is None else v)
        # brake for grip: more turn requested than the grip allows at this speed (never at full lock
        # within grip, so a slow hairpin keeps rolling)
        self.cornering = self.speed > 0 and abs(self.request) > self.steer_limit() + 1e-3
        if self.slipping and self.speed > 0:
            target, rate = 0.0, self.slip_decel   # stability control overrides the keys
        elif self.stopping:
            target, rate = 0.0, self.brake
        elif self.cornering and throttle >= 0:
            # brake until the requested turn has grip, at least the coast rate. Not capped by grip_left:
            # braking sheds speed, and with it lateral acceleration, faster than the friction circle would allow
            target, rate = 0.0, max(self.coast, self.corner_decel)
        elif not throttle:
            target, rate = 0.0, self.coast         # cruise (coast 0) or engine braking
        elif throttle > 0:
            if self.speed >= 0:
                # traction control: acceleration only from the grip the turn leaves
                target, rate = self.max_speed, min(self.accel, self.grip_left())
            else:
                target, rate = self.max_speed, self.brake
        else:
            target, rate = -self.max_reverse, (self.brake if self.speed > 0 else self.accel)
        self.speed = approach(self.speed, target, rate * dt)
        if self.stopping and self.speed == 0.0:
            self.stopping = False

        if self.slipping:
            self.request = self.steer          # more lock while sliding only slides more
        elif steer:
            if self.request * steer < 0:
                self.request = self.steer      # counter-steer from the wheels, not from a request above them
            self.request = approach(self.request, steer * self.max_steer, self.steer_rate * dt)
        else:
            self.request = approach(self.request, 0.0, self.center_rate * dt)
        limit = self.steer_limit()   # at the new speed
        self.steer = max(-limit, min(limit, self.request))
        return self.speed, self.steer


def main():
    if not sys.stdin.isatty():
        print('teleop_keyboard needs an interactive terminal (use ros2 run, not ros2 launch)')
        return

    # No rclpy SIGINT handler: Ctrl+C raises KeyboardInterrupt and the context stays valid
    # long enough to send the zero command.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(0))

    node = rclpy.create_node('teleop_keyboard')
    p = node.declare_parameter
    rate = p('rate', 20.0).value                      # Hz; adapter stops after cmd_timeout = 0.5 s
    vehicle = os.path.join(get_package_share_directory('control'), 'config', 'vehicle.yaml')
    with open(vehicle) as f:
        geometry = yaml.safe_load(f)['/**']['ros__parameters']
    car = Smoother(
        max_speed=p('max_speed', 3.0).value,                    # m/s
        max_reverse=p('max_reverse', 1.0).value,                # m/s
        accel=p('max_acceleration', 1.0).value,                 # m/s^2, W held (adapter ramps at 2.5)
        brake=p('brake_deceleration', 3.0).value,               # m/s^2, S held / E
        coast=p('coast_deceleration', 2.0).value,               # m/s^2 with no throttle key; 0 = cruise
        max_steer=p('max_steering_angle', geometry['max_steering_angle']).value,  # rad, control/config/vehicle.yaml
        steer_rate=p('steer_rate', 1.0).value,                  # rad/s, A/D held
        center_rate=p('center_rate', 2.0).value,                # rad/s, back to straight on release
        wheelbase=p('wheelbase', geometry['wheelbase']).value,  # m
        max_lat_accel=p('max_lateral_accel', 3.5).value,        # m/s^2; sim front grip saturates at ~3.5-4.4
        max_total_accel=p('max_total_accel', 4.0).value,        # m/s^2, traction control's grip budget
        corner_decel=p('corner_deceleration', 3.0).value,       # m/s^2, corner assist; 0 = off
        slip_decel=p('slip_deceleration', 4.0).value,           # m/s^2, stability control braking
        slip_ratio=p('slip_ratio', 0.85).value,                  # yaw rate / expected below this is sliding
        slip_min_yaw=p('slip_min_yaw_rate', 0.3).value,         # rad/s expected before slip is judged
        slip_confirm=p('slip_confirm_time', 0.15).value)         # s; the yaw rate lags a steering change ~0.1 s
    if not p('stability_control', True).value:
        car.slip_decel = 0.0
    imu_topic = p('imu_topic', '/sensors/imu/raw').value
    odom_topic = p('odom_topic', '/odom').value
    # macOS / X11 auto-repeat: first repeat after ~0.25-0.5 s, then every ~30-90 ms
    repeat_delay = p('key_repeat_delay', 0.5).value
    repeat_window = p('key_repeat_window', 0.15).value
    drive_pub = node.create_publisher(AckermannDriveStamped, '/drive', 10)
    reset_pub = node.create_publisher(Bool, '/autodrive/reset_command', 1)
    measured = {}   # 'yaw' / 'v' -> (value, time received)
    node.create_subscription(Imu, imu_topic,
                             lambda m: measured.update(yaw=(m.angular_velocity.z, time.monotonic())), 10)
    node.create_subscription(Odometry, odom_topic,
                             lambda m: measured.update(v=(m.twist.twist.linear.x, time.monotonic())), 10)

    def fresh(name, now):
        value, t = measured.get(name, (None, 0.0))
        return value if now - t < 0.3 else None

    def publish(speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        drive_pub.publish(msg)

    print(HELP.format(accel=car.accel, max_speed=car.max_speed, brake=car.brake, coast=car.coast, max_reverse=car.max_reverse,
                      max_steer=car.max_steer, max_lat=car.max_lat_accel, corner=car.corner_decel,
                      slip=car.slip_decel))
    fd = sys.stdin.fileno()
    settings = termios.tcgetattr(fd)
    held_until = {}   # key -> time its hold expires
    rest = ''         # an escape sequence cut off at the end of the last read
    shown = None
    period = 1.0 / rate
    last_t = next_t = time.monotonic()
    try:
        tty.setcbreak(fd)
        while True:
            ready, _, _ = select.select([fd], [], [], max(0.0, next_t - time.monotonic()))
            now = time.monotonic()
            # os.read, not sys.stdin.read: the buffered reader would hold back keys select can't see
            if ready:
                text, rest = split_escape(rest + os.read(fd, 32).decode(errors='ignore'))
                if handle_keys(text, now, held_until, car, repeat_delay, repeat_window):
                    reset_pub.publish(Bool(data=True))
            rclpy.spin_once(node, timeout_sec=0.0)   # gyro and odometry for stability control
            if now < next_t:
                continue

            def axis(pos, neg):
                return (held_until.get(pos, 0.0) >= now) - (held_until.get(neg, 0.0) >= now)

            yaw, v = fresh('yaw', now), fresh('v', now)
            speed, steer = car.step(now - last_t, axis('w', 's'), axis('a', 'd'),
                                    yaw_rate=yaw if car.slip_decel > 0 and v is not None else None, v=v)
            last_t = now
            publish(speed, steer)
            assist = 'SLIP' if car.slipping else 'GRIP' if car.cornering else '    '
            state = (round(speed, 2), round(steer, 3), round(car.steer_limit(), 3), assist)
            if state != shown:
                shown = state
                print(f'\rspeed {speed:+5.2f} m/s   steer {steer:+6.3f} rad (asked {car.request:+6.3f}, '
                      f'grip allows {car.steer_limit():.3f})  {assist}  ', end='', flush=True)
            next_t += period
            if next_t < now:   # fell behind, don't burst
                next_t = now + period
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)
        for _ in range(5):
            publish(0.0, 0.0)
            time.sleep(0.02)
        print('\nstopped, sent zero command')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

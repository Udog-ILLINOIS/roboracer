import math

import pytest

from autodrive_bridge.teleop_keyboard import Smoother, handle_keys, split_escape


@pytest.fixture
def car():
    return Smoother(max_speed=5.0, max_reverse=1.0, accel=1.5, brake=3.0, coast=1.0, max_steer=0.524,
                    steer_rate=1.0, center_rate=2.0, wheelbase=0.33, max_lat_accel=5.0,
                    max_total_accel=6.0, corner_decel=1.0)


def run(car, seconds, throttle, steer, dt=0.05):
    for _ in range(round(seconds / dt)):
        car.step(dt, throttle, steer)


def test_throttle_ramps_then_coasts(car):
    run(car, 1.0, 1, 0)
    assert car.speed == pytest.approx(1.5)
    run(car, 1.0, 0, 0)
    assert car.speed == pytest.approx(0.5)       # coasts down at 1 m/s^2
    run(car, 1.0, 0, 0)
    assert car.speed == 0.0                      # and stops without reversing
    run(car, 10.0, 1, 0)
    assert car.speed == pytest.approx(5.0)


def test_brake_then_reverse(car):
    car.speed = 3.0
    run(car, 1.0, -1, 0)
    assert car.speed == pytest.approx(0.0)
    run(car, 2.0, -1, 0)
    assert car.speed == pytest.approx(-1.0)


def test_e_stops_without_reversing(car):
    car.speed = 3.0
    car.stopping = True
    run(car, 2.0, 0, 0)
    assert car.speed == 0.0 and not car.stopping


def test_steering_ramps_and_recentres(car):
    run(car, 0.25, 0, 1)
    assert car.steer == pytest.approx(0.25)
    run(car, 1.0, 0, 1)
    assert car.steer == pytest.approx(0.524)     # full lock when stopped
    run(car, 0.5, 0, 0)
    assert car.steer == 0.0


def test_lock_keeps_lateral_accel_under_limit(car):
    car.corner_decel = car.coast = 0.0
    car.speed = 4.0
    run(car, 2.0, 0, -1)
    lat = car.speed ** 2 * math.tan(abs(car.steer)) / 0.33
    assert car.steer < 0 and lat == pytest.approx(5.0)


def test_traction_control_cuts_accel_in_a_corner(car):
    car.coast = car.corner_decel = car.center_rate = 0.0
    car.speed = 3.0
    car.request = car.steer = 0.05                # turning, within grip: ay = 9 * tan(0.05) / 0.33 = 1.4
    run(car, 0.2, 1, 0)                           # A released, wheels held turned: W alone
    assert car.speed == pytest.approx(3.0 + 1.5 * 0.2)   # a 6 m/s^2 budget leaves more than accel 1.5
    car.speed, car.max_total_accel = 3.0, 1.6     # leaves sqrt(1.6^2 - 1.4^2) ~ 0.8
    run(car, 0.2, 1, 0)
    assert 3.0 + 0.1 < car.speed < 3.0 + 0.2


def test_brakes_until_the_requested_turn_has_grip(car):
    car.coast = 0.0
    car.speed = 4.0
    run(car, 1.0, 0, 1)                           # request passes the grip limit in ~0.1 s, then brakes
    assert 3.0 < car.speed < 3.2 and car.cornering
    run(car, 3.0, 0, 1)                           # full lock requested: brakes until full lock has grip
    assert car.steer == pytest.approx(0.524) and not car.cornering
    assert car.speed == pytest.approx(math.sqrt(0.33 * 5.0 / math.tan(0.524)), abs=0.06)
    run(car, 1.0, 0, 0)
    v = car.speed
    run(car, 1.0, 0, 0)
    assert car.speed == v                         # released: no braking, cruise


def test_short_press_brakes_less_than_long_hold(car):
    car.coast = 0.0
    car.speed = 3.0
    run(car, 0.2, 0, 1)
    run(car, 1.0, 0, 0)
    short = car.speed
    car.stop()
    car.speed = 3.0
    run(car, 0.6, 0, 1)
    run(car, 1.0, 0, 0)
    assert 3.0 > short > car.speed


def test_wheels_never_exceed_grip(car):
    car.coast = car.corner_decel = 0.0            # no braking: the request stays above the grip
    car.speed = 4.0
    run(car, 2.0, 0, -1)
    assert car.request == pytest.approx(-0.524) and car.steer == pytest.approx(-car.steer_limit())


def test_countersteer_starts_from_the_wheels(car):
    car.coast = car.corner_decel = 0.0
    car.speed = 4.0
    run(car, 2.0, 0, 1)                           # request at full lock, wheels at ~0.1 rad
    run(car, 0.2, 0, -1)
    assert car.steer < 0.0                        # unwinding the request first would leave it at +0.32


def test_corner_assist_overrides_throttle(car):
    car.speed = 4.0
    run(car, 1.0, 1, 1)
    assert car.speed < 3.3 and car.cornering    # 4 -> 4.15 while the lock builds, then 1 m/s^2 down


def slide(car, seconds, ratio, throttle=1, dt=0.05):
    """Hold A with the gyro reporting `ratio` of the yaw rate the steering asks for."""
    for _ in range(round(seconds / dt)):
        expected = car.speed * math.tan(car.steer) / 0.33
        car.step(dt, throttle, 1, yaw_rate=ratio * expected, v=car.speed)


def test_stability_control_brakes_a_slide(car):
    car.corner_decel = car.coast = 0.0
    car.speed = 3.0
    run(car, 1.0, 0, 1)
    steer = car.steer
    slide(car, 0.1, 0.5)
    assert not car.slipping                       # not confirmed yet (transient lag)
    slide(car, 0.5, 0.5)
    assert car.slipping
    assert car.speed == pytest.approx(3.0 - 4.0 * 0.5)    # 10 steps from the 0.15 s confirm on, W ignored
    assert car.steer <= steer + 1e-9             # no extra lock while sliding


def test_stability_control_quiet_when_gripping(car):
    car.corner_decel = car.coast = 0.0
    car.speed = 3.0
    run(car, 1.0, 0, 1)
    slide(car, 1.0, 0.95, throttle=0)
    assert not car.slipping and car.speed == pytest.approx(3.0)


def test_corner_assist_lets_a_slow_hairpin_roll(car):
    car.coast = 0.0
    car.speed = 1.0                               # full lock is within grip at 1 m/s
    run(car, 2.0, 0, 1)
    assert car.steer == pytest.approx(0.524) and car.speed == 1.0 and not car.cornering


def keys(text, now, held, car):
    return handle_keys(text, now, held, car, repeat_delay=0.5, repeat_window=0.15)


def test_opposite_key_replaces_the_held_one(car):
    held = {}
    keys('w', 0.0, held, car)
    keys('s', 0.1, held, car)            # within W's 0.5 s hold: brakes, not W - S = coast
    assert 'w' not in held and held['s'] == pytest.approx(0.6)
    keys('a', 0.2, held, car)
    keys('d', 0.3, held, car)            # D after an A tap steers right, not straight
    assert 'a' not in held and 'd' in held and 's' in held


def test_arrow_keys_are_not_letters(car):
    held = {}
    assert not keys('\x1b[A\x1b[D\x1bOA', 0.0, held, car)     # up, left (normal and application mode)
    assert held == {}
    keys('\x1b[Bw', 0.0, held, car)     # a letter after an arrow still counts
    assert set(held) == {'w'}
    assert keys('R', 0.0, held, car) and held == {}


def test_modified_arrow_keys_are_not_letters(car):
    held = {}
    # Ctrl+Left, Shift+Left, Alt+Up, Ctrl+Up: ESC [ 1 ; m A-D, the 'D' / 'A' would steer
    assert not keys('\x1b[1;5D\x1b[1;2D\x1b[1;3A\x1b[1;5A\x1b[3~', 0.0, held, car)
    assert held == {}


def test_escape_cut_off_between_reads(car):
    text, rest = split_escape('w\x1b[1;5')
    assert (text, rest) == ('w', '\x1b[1;5')
    text, rest = split_escape(rest + 'Ds')      # the rest of Ctrl+Left, then S
    assert (text, rest) == ('\x1b[1;5Ds', '')
    held = {}
    keys(text, 0.0, held, car)
    assert set(held) == {'s'}
    assert split_escape('\x1b') == ('', '\x1b') and split_escape('\x1bO') == ('', '\x1bO')

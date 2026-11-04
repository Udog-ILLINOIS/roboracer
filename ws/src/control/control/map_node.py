"""ROS 2 MAP controller; /odom linear.x is forward speed, with heading as velocity direction."""
import json
import math
import os
import time
import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from rclpy.parameter import Parameter
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory
from perception.tracks import track_dir
from .map_guidance import Trajectory,l1_acceleration
from .map_lut import Lookup
from .input_health import InputHealth
from std_msgs.msg import String

class MAP(Node):
    def __init__(self):
        super().__init__('map_node')
        p=self.declare_parameter
        self.speed_scale=p('speed_scale',.5).value
        self.max_speed=p('max_speed',0.).value
        self.slope=p('lookahead_slope',.4).value
        self.offset=p('lookahead_offset',0.).value
        self.lmin=p('lookahead_min',.6).value
        self.lmax=p('lookahead_max',3.).value
        self.pose_timeout=p('pose_timeout',.5).value
        self.odom_timeout=p('odom_timeout',.5).value
        self.max_offset=p('max_offset',3.).value
        self.max_jump=p('max_pose_jump',2.).value
        self.rate=p('steering_rate_limit',3.2).value
        # Command at most lut top - margin; measured speed may overshoot the top column by up to overshoot.
        self.lut_margin=p('lut_speed_margin',.3).value
        self.lut_overshoot=p('lut_speed_overshoot',.5).value
        self.reverse_tolerance=p('reverse_speed_tolerance',.05).value
        self.health=InputHealth(
            recovery_seconds=p('recovery_seconds',.25).value,
            recovery_samples=p('recovery_samples',3).value,
            heading_tolerance=p('heading_consistency_tolerance',.25).value,
            clock_tolerance=p('clock_jump_tolerance',.25).value,
            reseed_seconds=p('heading_reseed_seconds',1.).value)
        self.lock=p('max_steering_angle',Parameter.Type.DOUBLE).value
        self.wheelbase=p('wheelbase',Parameter.Type.DOUBLE).value
        hz=p('update_rate',40.).value
        numeric=[self.speed_scale,self.max_speed,self.slope,self.offset,self.lmin,self.lmax,self.pose_timeout,self.odom_timeout,self.max_offset,self.max_jump,self.rate,hz,self.lock,self.wheelbase,self.lut_margin,self.lut_overshoot,self.reverse_tolerance]
        if any(v is None or not isinstance(v,(int,float)) or not math.isfinite(v) for v in numeric) or min(self.speed_scale,self.lmin,self.pose_timeout,self.odom_timeout,self.max_offset,self.max_jump,self.rate,hz,self.lock,self.wheelbase)<=0 or self.lmax<self.lmin or self.max_speed<0 or min(self.lut_margin,self.lut_overshoot,self.reverse_tolerance)<0:
            raise ValueError('invalid MAP configuration or missing vehicle.yaml')
        share=get_package_share_directory('control')
        model_path=p('model_path',os.path.join(share,'config','map_model.json')).value
        lut_path=p('lut_path',os.path.join(share,'config','autodrive_map_lut.npz')).value
        with open(model_path) as f: model=json.load(f)
        if model['model'] not in ('linear','pacejka') or abs(model['lf']+model['lr']-self.wheelbase)>1e-6:
            raise ValueError('model/wheelbase mismatch')
        self.lut=Lookup(lut_path,model)
        self.v_cap=float(self.lut.speeds[-1])-self.lut_margin
        if self.v_cap<=self.lut.speeds[0]: raise ValueError('lut_speed_margin leaves no usable LUT speed range')
        self.cap=min(self.max_speed if self.max_speed>0 else float('inf'),self.v_cap)
        path=p('raceline_path','').value or os.path.join(track_dir(p('track','').value),'raceline.csv')
        self.trajectory=Trajectory(np.loadtxt(path,delimiter=';',ndmin=2),p('max_segment',1.).value,p('search_window',2.).value,p('reacquire_distance',1.).value)
        self.pose=None; self.odom=None; self.last_xy=None; self.last_stamp=None; self.last_steer=0.; self.last_command_time=None
        self.still_xy=None; self.still_since=None
        self.last_steer_time=time.monotonic(); self.odom_received=None
        self.pub=self.create_publisher(AckermannDriveStamped,'/drive',10)
        self.create_subscription(PoseStamped,p('pose_topic','/localized_pose').value,self.on_pose,10)
        self.create_subscription(Odometry,p('odom_topic','/odom').value,self.on_odom,10)
        reset_topic=p('sim_reset_topic','/autodrive/roboracer_1/sim_reset').value
        if reset_topic:
            self.create_subscription(String,reset_topic,lambda msg:self.reset(),10)
        self.create_timer(1/hz,self.tick)

    def reset(self):
        self.trajectory.reset(); self.last_xy=None; self.last_stamp=None; self.last_steer=0.; self.last_command_time=None
        self.still_xy=None; self.still_since=None
        self.last_steer_time=time.monotonic(); self.odom_received=None; self.health.reset()
        self.pose=None; self.odom=None

    def on_pose(self,msg): self.pose=msg
    def on_odom(self,msg):
        self.odom=msg
        self.odom_received=time.monotonic()

    def publish(self,speed=0.,steer=0.):
        msg=AckermannDriveStamped()
        msg.header.stamp=self.get_clock().now().to_msg(); msg.header.frame_id='base_link'
        msg.drive.speed=float(speed); msg.drive.steering_angle=float(steer)
        self.pub.publish(msg)

    def stop(self,reason,invalidate=True):
        self.get_logger().warning('MAP stop: '+reason,throttle_duration_sec=1.)
        # Brake while retaining steering, and retain the slew reference on resume.
        self.publish(0.,self.last_steer)
        self.last_steer_time=time.monotonic()
        if invalidate: self.health.fault()

    def standstill(self,x,y,speed,steady_now):
        """True once wheel speed AND the pose position have both been still for heading_reseed_seconds.

        Wheel odometry alone is not enough (in the recorded hairpin it read 0.03 m/s at 1 m/s), so the
        localizer's position must agree. Only then may the pose-jump, heading and time-reversal latches
        take the current pose as their new reference (e.g. after the localizer relocated a stopped car).
        """
        if abs(speed)>=.1:
            self.still_xy=self.still_since=None
            return False
        if self.still_xy is None or math.hypot(x-self.still_xy[0],y-self.still_xy[1])>.05:
            self.still_xy=(x,y); self.still_since=steady_now
            return False
        return steady_now-self.still_since>=self.health.reseed_seconds

    def heading_plausible(self,x,y,yaw):
        # A re-seeded heading must also follow the raceline near the car: never adopt a reversed or
        # sideways heading (the hairpin's 119 deg error) just because it is steady.
        i=int(np.argmin(np.linalg.norm(self.trajectory.xy-[x,y],axis=1)))
        return math.cos(yaw-self.trajectory.heading[i])>math.cos(1.)

    def tick(self):
        start=time.perf_counter()
        try: self._tick()
        except (ValueError,IndexError,FloatingPointError,OverflowError) as exc:
            self.stop(str(exc))
        finally:
            elapsed=(time.perf_counter()-start)*1000
            self.get_logger().debug(f'MAP computation {elapsed:.2f} ms',throttle_duration_sec=1.)

    def _tick(self):
        now=self.get_clock().now().nanoseconds*1e-9
        steady_now=time.monotonic()
        clock_fault=self.health.advance(now,steady_now,
            self.odom.twist.twist.angular.z if self.odom is not None else 0.,
            self.odom_received is not None and steady_now-self.odom_received<.3)
        if clock_fault: return self.stop(clock_fault,invalidate=False)
        if self.pose is None or self.odom is None: return self.stop('missing pose or odometry')
        pose=self.pose; odom=self.odom
        ps=pose.header.stamp.sec+pose.header.stamp.nanosec*1e-9
        os_=odom.header.stamp.sec+odom.header.stamp.nanosec*1e-9
        if (ps<=0 or os_<=0 or now-ps>self.pose_timeout or now-os_>self.odom_timeout or ps-now>.1 or os_-now>.1):
            return self.stop('stale or future input')
        if pose.header.frame_id!='map': return self.stop('invalid pose frame')
        q=pose.pose.orientation
        quat=np.array([q.x,q.y,q.z,q.w]); x=pose.pose.position.x; y=pose.pose.position.y
        speed=odom.twist.twist.linear.x
        if not np.isfinite([x,y,speed,*quat]).all() or abs(np.linalg.norm(quat)-1)>.05 or speed<-self.reverse_tolerance:
            return self.stop('nonfinite or invalid state')
        still=self.standstill(x,y,speed,steady_now)
        if self.last_stamp is not None and (ps<self.last_stamp or (self.last_command_time is not None and now<self.last_command_time)):
            if not still: return self.stop('time reversal')
            self.last_stamp=self.last_command_time=None  # time source restarted while stopped: new reference
            return self.stop('time reference reset at standstill, waiting for recovery')
        if self.last_xy is not None and math.hypot(x-self.last_xy[0],y-self.last_xy[1])>self.max_jump:
            if not still: return self.stop('implausible pose jump')
            self.last_xy=(x,y)  # relocated while stopped: the steady pose is the new reference
            return self.stop('pose re-anchored at standstill, waiting for recovery')
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        rejection=self.health.accept(yaw,(ps,os_),steady_now,stationary=still and self.heading_plausible(x,y,yaw))
        if rejection: return self.stop(rejection,invalidate=False)
        # Use the local waypoint for lookahead, then the target waypoint for commanded speed.
        closest=self.trajectory.closest
        local_v=self.trajectory.rows[closest,5] if closest is not None else self.trajectory.rows[0,5]
        command_v=min(local_v*self.speed_scale,self.cap)
        lookahead=float(np.clip(self.slope*command_v+self.offset,self.lmin,self.lmax))
        target,distance,eta,error=self.trajectory.target(x,y,yaw,lookahead,self.max_offset)
        command_v=min(self.trajectory.rows[target,5]*self.speed_scale,self.cap)
        if speed>self.lut.speeds[-1]+self.lut_overshoot: return self.stop('measured speed exceeds LUT')
        effective=float(np.clip(speed,self.lut.speeds[0],self.lut.speeds[-1]))
        accel=l1_acceleration(effective,eta,distance)
        steer,clamped,column=self.lut.query(accel,effective)
        saturated=abs(steer)>self.lock
        steer=float(np.clip(steer,-self.lock,self.lock))
        steering_dt=max(0.,min(steady_now-self.last_steer_time,.1))
        steer=float(np.clip(steer,self.last_steer-self.rate*steering_dt,self.last_steer+self.rate*steering_dt))
        self.publish(command_v,steer)
        self.last_steer=steer; self.last_steer_time=steady_now; self.last_command_time=now; self.last_stamp=ps; self.last_xy=(x,y)
        self.get_logger().info(f'MAP v={speed:.2f} cmd={command_v:.2f} target={target} L={lookahead:.2f} a={accel:.2f} steer={steer:.3f} error={error:.2f} low_speed={speed<self.lut.speeds[0]} clamp={clamped} saturation={saturated} column={column}',throttle_duration_sec=1.)

def main():
    # Handle Ctrl+C ourselves (as pure_pursuit_node): rclpy's default handler shuts ROS down first,
    # so the final stop could never be published.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=MAP()
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.1)
    except KeyboardInterrupt:
        pass
    for _ in range(10):  # 0.5 s of stops so the adapter brakes
        node.publish(0.,node.last_steer)
        time.sleep(.05)
    node.destroy_node()
    rclpy.try_shutdown()

#!/usr/bin/env python3
"""
IBVSControllerNode — Image-Based Visual Servoing for MoveIt Servo.

Architecture:
  - Subscribes to /yolo/target_bounding_box (Float32MultiArray: [cx, cy, w, h])
  - Subscribes to /ibvs/enable (Bool) to gate output
  - Publishes TwistStamped to /servo_node/delta_twist_cmds at 50 Hz

Frame convention (servo.yaml: robot_link_command_frame = base_link):
  MoveIt Servo interprets the twist in the frame set by robot_link_command_frame.
  With the camera mounted eye-in-hand on link6, looking forward along the gripper:
    - Object is LEFT  of centre  -> arm must rotate CCW around base Z -> vel_z_angular > 0
    - Object is RIGHT of centre  -> arm must rotate CW  around base Z -> vel_z_angular < 0
  For simplicity we keep the approach in Cartesian linear:
    - Image error_x (pixels, right = positive in camera) maps to base_link -Y (arm yaw left).
    - Image error_y (pixels, down  = positive in camera) maps to base_link -Z (arm pitches down).
    - Depth error (sqrt area) maps to base_link +X (approach forward).

  All three axes are published simultaneously every cycle so MoveIt Servo has a
  continuous, fresh command stream and never times out.

Fixes vs. old code:
  1. Time 0 stamp    — Uses rclpy.time.Time() (zero) so tf2 always returns the latest
                        transform without ExtrapolationException dropouts.
  2. Correct frame   — Published in 'base_link' (matching robot_link_command_frame in
                        servo.yaml). The mapping below accounts for the camera → base
                        axis relationship: image-x error → base -Y, image-y → base -Z.
  3. Sqrt depth math — Depth velocity is proportional to
                        (sqrt(target_area) - sqrt(current_area)), not raw area diff.
  4. No zero spam    — When IBVS is disabled or the target is lost the timer is simply
                        not firing; we publish nothing rather than a flood of zeros.
  5. No deadbands    — Proportional control; every pixel error produces a velocity.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import TwistStamped


class IBVSControllerNode(Node):

    def __init__(self):
        super().__init__('ibvs_controller_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('target_pixel_x',          320.0)
        self.declare_parameter('target_pixel_y',          240.0)
        self.declare_parameter('target_area_px',        50000.0)  # pixels²
        self.declare_parameter('gain_lateral',            0.0003)  # m/s per px  (image-x → base -Y)
        self.declare_parameter('gain_vertical',           0.0003)  # m/s per px  (image-y → base -Z)
        self.declare_parameter('gain_depth',              0.0020)  # m/s per sqrt-px-unit
        self.declare_parameter('max_linear_speed',          0.06)  # m/s  hard cap
        self.declare_parameter('detection_timeout',          0.5)  # s — silence = target lost
        self.declare_parameter('command_frame',        'base_link')

        self.tgt_x    = float(self.get_parameter('target_pixel_x').value)
        self.tgt_y    = float(self.get_parameter('target_pixel_y').value)
        self.tgt_area = float(self.get_parameter('target_area_px').value)
        self.tgt_sqrt = math.sqrt(max(self.tgt_area, 1.0))

        self.gain_lat  = float(self.get_parameter('gain_lateral').value)
        self.gain_vert = float(self.get_parameter('gain_vertical').value)
        self.gain_dep  = float(self.get_parameter('gain_depth').value)
        self.v_max     = float(self.get_parameter('max_linear_speed').value)
        self.det_timeout = float(self.get_parameter('detection_timeout').value)
        self.cmd_frame = str(self.get_parameter('command_frame').value)

        # ── State ─────────────────────────────────────────────────────────────
        self._active           = False
        self._last_detect_time = 0.0        # wall clock
        self._vx = self._vy = self._vz = 0.0

        # ── Comms ─────────────────────────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray, '/yolo/target_bounding_box', self._bbox_cb, 10)
        self.create_subscription(
            Bool, '/ibvs/enable', self._enable_cb, 10)

        self._twist_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # 50 Hz publish timer — started/stopped based on enable flag
        self._timer = self.create_timer(0.02, self._publish_cb)
        self._timer.cancel()   # start paused; enabled via /ibvs/enable

        self.get_logger().info(
            f'IBVSControllerNode ready — frame={self.cmd_frame}, '
            f'target=({self.tgt_x:.0f},{self.tgt_y:.0f})px, '
            f'target_area={self.tgt_area:.0f}px²'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _enable_cb(self, msg: Bool) -> None:
        if msg.data and not self._active:
            self._active = True
            self._timer.reset()           # restart 50 Hz loop
            self.get_logger().info('IBVS ENABLED — publishing twist commands')
        elif not msg.data and self._active:
            self._active = False
            self._timer.cancel()          # stop publishing; no zero spam
            self._vx = self._vy = self._vz = 0.0
            self.get_logger().info('IBVS DISABLED — twist output paused')

    def _bbox_cb(self, msg: Float32MultiArray) -> None:
        """Compute proportional Cartesian velocities from 2D bounding-box error."""
        if not self._active or len(msg.data) < 4:
            return

        cx, cy, w, h = (float(v) for v in msg.data[:4])
        self._last_detect_time = time.monotonic()

        # ── Pixel errors (positive = target is to the right / below centre) ──
        err_x = cx - self.tgt_x   # + means target right of centre
        err_y = cy - self.tgt_y   # + means target below centre

        # ── Depth error via sqrt of bounding-box area ─────────────────────────
        # sqrt(area) is approximately proportional to linear size (1/depth),
        # so (tgt_sqrt - cur_sqrt) grows linearly with distance error.
        cur_area = w * h
        cur_sqrt = math.sqrt(max(cur_area, 1.0))
        err_depth = self.tgt_sqrt - cur_sqrt   # + means too far → move forward

        # ── Velocity mapping (camera eye-in-hand, looking forward) ────────────
        # base_link axes:  X=forward, Y=left, Z=up
        #
        # Camera image axes relative to base_link at nominal pose:
        #   image  +x  →  base  -Y  (target right of centre → move arm right → -Y)
        #   image  +y  →  base  -Z  (target below centre    → move arm down  → -Z)
        #   depth  +   →  base  +X  (target too far away    → move arm forward)
        vx = self._clamp( self.gain_dep  *  err_depth,  self.v_max)
        vy = self._clamp(-self.gain_lat  *  err_x,      self.v_max)
        vz = self._clamp(-self.gain_vert *  err_y,      self.v_max)

        self._vx, self._vy, self._vz = vx, vy, vz

    def _publish_cb(self) -> None:
        """Publish the latest velocity command at 50 Hz."""
        # If we haven't seen a detection recently, stop the arm.
        if time.monotonic() - self._last_detect_time > self.det_timeout:
            # Target lost — publish one zero to halt servo, then let timer idle
            # (we do NOT flood zeros every cycle; MoveIt Servo has its own timeout)
            if self._vx != 0.0 or self._vy != 0.0 or self._vz != 0.0:
                self._vx = self._vy = self._vz = 0.0
                self._send_twist(0.0, 0.0, 0.0)
                self.get_logger().warn(
                    'Target lost — sent one zero-twist to halt servo',
                    throttle_duration_sec=2.0)
            return

        self._send_twist(self._vx, self._vy, self._vz)
        self.get_logger().info(
            f'[Twist] Vx={self._vx:+.4f} Vy={self._vy:+.4f} Vz={self._vz:+.4f} m/s',
            throttle_duration_sec=1.0,
        )

    def _send_twist(self, vx: float, vy: float, vz: float) -> None:
        msg = TwistStamped()
        # Time(0) — fetch the *latest* available TF transform; avoids
        # ExtrapolationException when the servo node looks up the frame.
        msg.header.stamp    = rclpy.time.Time().to_msg()
        msg.header.frame_id = self.cmd_frame
        msg.twist.linear.x  = vx
        msg.twist.linear.y  = vy
        msg.twist.linear.z  = vz
        # No angular component — pure Cartesian approach
        self._twist_pub.publish(msg)

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))


def main(args=None):
    rclpy.init(args=args)
    node = IBVSControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

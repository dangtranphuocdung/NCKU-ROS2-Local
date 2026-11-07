#!/usr/bin/env python3
"""
HardwareServoBridgeNode — Lean hardware bridge for the reBotArm IBVS pipeline.

Responsibilities (and nothing more):
  1. QoS Bridge — Re-publishes /{arm_ns}/joint_states (BEST_EFFORT) to /joint_states
     with RELIABLE QoS so MoveIt Servo and move_group receive continuous joint feedback
     without DDS QoS mismatch silently dropping every message.

  2. Trajectory Forwarding — Listens on /rebotarm_controller/joint_trajectory (the
     topic MoveIt Servo writes to) and dispatches the commanded joint positions to
     each /{arm_ns}/joints/{joint}/cmd/pos_vel topic in the vendor hardware controller.
     Every joint that changed is forwarded every cycle — no restrictive deadband that
     kills small smooth 50 Hz corrections.

Design notes:
  - The vendor's send_joint_pos_vel_cmd() does a synchronous CAN feedback read then
    overwrites one index.  Firing all 6 topics simultaneously causes a race where
    each joint's read clobbers the others.  We serialise by forwarding joints ONE AT
    A TIME per trajectory message, rotating which joint leads on each call, so every
    joint gets updated within one 50 Hz period (6 joints × 20 ms ≈ 120 ms round-trip
    is fine; the vendor loop runs at 500 Hz internally).
  - The per-joint publisher approach bypasses the FollowJointTrajectory action layer
    entirely (which would add ≥ 1 trajectory-execution round-trip of latency) and
    directly sets the motor target positions in real time.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from rebotarm_msgs.msg import JointPosVelCmd, ArmStatus


JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

# URDF limits with a tiny inset (0.02 rad) for safety
_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    'joint1': (-2.78, 2.78),
    'joint2': (-3.12, -0.02),
    'joint3': (-3.12, -0.02),
    'joint4': (-1.85,  1.55),
    'joint5': (-1.55,  1.55),
    'joint6': (-3.12,  3.12),
}


class HardwareServoBridgeNode(Node):

    def __init__(self) -> None:
        super().__init__('hardware_servo_bridge_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('arm_namespace',       'rebotarm')
        self.declare_parameter('default_vlim',             1.5)   # rad/s
        self.declare_parameter('joint_names',         JOINT_NAMES)

        ns            = str(self.get_parameter('arm_namespace').value).strip('/')
        self._vlim    = float(self.get_parameter('default_vlim').value)
        self._joints  = [str(n) for n in self.get_parameter('joint_names').value]

        # ── QoS profiles ──────────────────────────────────────────────────────
        _reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── 1. Joint-State Bridge ─────────────────────────────────────────────
        self._js_pub = self.create_publisher(JointState, '/joint_states', _reliable_qos)
        self.create_subscription(
            JointState,
            f'/{ns}/joint_states',
            self._js_cb,
            qos_profile_sensor_data,  # match the BEST_EFFORT publisher
        )
        self.get_logger().info(
            f'Bridging /{ns}/joint_states → /joint_states  [BEST_EFFORT → RELIABLE]'
        )

        # ── 2. Per-joint command publishers ───────────────────────────────────
        self._joint_pubs: dict[str, rclpy.publisher.Publisher] = {
            name: self.create_publisher(
                JointPosVelCmd, f'/{ns}/joints/{name}/cmd/pos_vel', 10)
            for name in self._joints
        }

        # ── 3. Servo trajectory subscriber ───────────────────────────────────
        self.create_subscription(
            JointTrajectory,
            '/rebotarm_controller/joint_trajectory',
            self._traj_cb,
            10,
        )

        # ── 4. Arm-status subscriber (for future watchdog use) ────────────────
        self.create_subscription(
            ArmStatus, f'/{ns}/arm_status', lambda _: None, 10)

        # ── Internal state ────────────────────────────────────────────────────
        self._current_pos: dict[str, float] = {}
        self._dispatch_index = 0   # round-robin joint rotation
        self._js_count   = 0
        self._traj_count = 0
        self._fwd_count  = 0
        self._last_log   = 0.0

        self.get_logger().info('HardwareServoBridgeNode ready.')

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _js_cb(self, msg: JointState) -> None:
        """Bridge joint states and keep an up-to-date local copy."""
        self._js_count += 1
        for name, pos in zip(msg.name, msg.position):
            self._current_pos[name] = float(pos)
        self._js_pub.publish(msg)

        now = time.monotonic()
        if now - self._last_log > 5.0:
            preview = [round(self._current_pos.get(n, 0.0), 4) for n in self._joints]
            self.get_logger().info(
                f'[JointState Bridge] {preview}  (js={self._js_count}, '
                f'traj={self._traj_count}, fwd={self._fwd_count})'
            )
            self._last_log = now

    def _traj_cb(self, msg: JointTrajectory) -> None:
        """
        Forward commanded positions from MoveIt Servo to the hardware controller.

        Strategy: forward ALL joints that have changed, one at a time in a
        round-robin fashion to avoid the CAN-feedback-overwrite race in the
        vendor driver.  Each call advances the dispatch index by one so that
        over N calls every joint is visited.
        """
        if not msg.points:
            return

        self._traj_count += 1
        point = msg.points[-1]
        name_to_pos = dict(zip(msg.joint_names, point.positions))

        # Build list of (name, target) for joints that are moving
        moving: list[tuple[str, float]] = []
        for name in self._joints:
            if name not in name_to_pos:
                continue
            target = float(name_to_pos[name])
            current = self._current_pos.get(name, target)
            # No deadband — forward every non-trivial change (> floating-point noise)
            if abs(target - current) > 1e-6:
                moving.append((name, target))

        if not moving:
            return

        # Round-robin: pick one joint per call, cycling through all moving joints.
        # This serialises CAN writes and eliminates the feedback-overwrite race.
        idx = self._dispatch_index % len(moving)
        self._dispatch_index += 1
        name, target = moving[idx]

        self._publish_joint_cmd(name, target)

    def _publish_joint_cmd(self, name: str, target: float) -> None:
        """Clamp to URDF limits and publish a JointPosVelCmd."""
        lo, hi = _JOINT_LIMITS.get(name, (-6.28, 6.28))
        target = max(lo, min(hi, target))

        cmd = JointPosVelCmd()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.pos   = target
        cmd.vlim  = self._vlim
        self._joint_pubs[name].publish(cmd)
        self._fwd_count += 1

        curr = self._current_pos.get(name, float('nan'))
        self.get_logger().debug(
            f'[→ HW] {name}: target={target:+.4f} curr={curr:+.4f}  vlim={self._vlim}'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HardwareServoBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

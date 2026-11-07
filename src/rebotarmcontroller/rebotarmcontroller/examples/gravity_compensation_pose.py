#!/usr/bin/env python3
"""Start internal gravity compensation on the ROS controller and print end-effector XYZ & RPY in real time.

Provides lead-through teaching: manually move the arm while observing the end-effector
position (x, y, z) and orientation (roll, pitch, yaw in degrees and radians, plus quaternion),
computed via forward kinematics using the SDK in third_party/reBotArm_control_py.

Press [SPACE] at any time to record the current pose into a plaintext file in a designated folder.
"""

from __future__ import annotations

import argparse
import datetime
import json
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_sdk_in_syspath() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        sdk_dir = parent / "third_party" / "reBotArm_control_py"
        if (sdk_dir / "reBotArm_control_py").is_dir():
            sdk_str = str(sdk_dir)
            if sdk_str not in sys.path:
                sys.path.insert(0, sdk_str)
            return


_ensure_sdk_in_syspath()

import pinocchio as pin  # noqa: E402
from reBotArm_control_py.kinematics import (  # noqa: E402
    compute_fk,
    get_end_effector_frame,
    get_joint_names,
    load_robot_model,
    pad_q_for_model,
)

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rclpy.signals import SignalHandlerOptions  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402


def _call_trigger(
    node: Node,
    client,
    label: str,
    timeout_sec: float = 5.0,
) -> bool:
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(f"{label} service not available")
        return False
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done():
        node.get_logger().error(f"{label} timed out")
        return False
    result = future.result()
    if result is None or not result.success:
        message = result.message if result is not None else "no response"
        node.get_logger().error(f"{label} failed: {message}")
        return False
    node.get_logger().info(message if (message := result.message) else f"{label} OK")
    return True


class GravityCompensationPoseNode(Node):
    """ROS node to monitor joint states, stream FK pose, and save waypoints on spacebar."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("gravity_compensation_pose")
        self.namespace = args.namespace.strip("/")
        self.print_interval = 1.0 / max(0.1, float(args.rate))
        self.compact = bool(args.compact)
        self.change_only = bool(args.change_only)
        self.pos_threshold = float(args.pos_threshold)
        self.rot_threshold = float(args.rot_threshold)

        # Setup save folder and output file paths
        self.save_dir = Path(args.save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_file_txt = (
            self.save_dir / args.save_file
            if args.save_file
            else self.save_dir / f"poses_{session_id}.txt"
        )
        self.save_file_json = (
            self.save_file_txt.with_suffix(".json")
        )
        self.waypoint_count = 0
        self.saved_waypoints: list[dict[str, Any]] = []

        # Load Pinocchio kinematics model from third_party SDK
        self.model = load_robot_model()
        self.ee_frame = args.ee_frame if args.ee_frame else get_end_effector_frame()
        self.model_joint_names = get_joint_names(self.model)

        self.last_print_time = 0.0
        self.last_pos: np.ndarray | None = None
        self.last_rpy_deg: np.ndarray | None = None
        self.latest_pose_data: dict[str, Any] | None = None
        self.latest_pose_str: str = ""

        # Write session header to file
        self._init_save_file()

        self._sub = self.create_subscription(
            JointState,
            f"/{self.namespace}/joint_states",
            self._joint_state_cb,
            qos_profile_sensor_data,
        )

    def _init_save_file(self) -> None:
        header = (
            f"# ======================================================================\n"
            f"# reBotArm Saved Poses / Waypoints\n"
            f"# Session started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Model: {self.model.name} | End-effector frame: {self.ee_frame}\n"
            f"# ======================================================================\n\n"
        )
        with open(self.save_file_txt, "a", encoding="utf-8") as f:
            f.write(header)

    def _joint_state_cb(self, msg: JointState) -> None:
        now = time.monotonic()

        # Map received joint positions by name to match Pinocchio model order
        pos_dict = dict(zip(msg.name, msg.position))
        q_model = np.array(
            [pos_dict.get(name, 0.0) for name in self.model_joint_names],
            dtype=np.float64,
        )
        q_padded = pad_q_for_model(self.model, q_model)

        # Compute Forward Kinematics via third_party SDK
        position, rotation, _ = compute_fk(
            self.model, q_padded, frame_name=self.ee_frame
        )
        rpy_rad = pin.rpy.matrixToRpy(rotation)
        rpy_deg = np.degrees(rpy_rad)
        quat = pin.Quaternion(rotation)

        arm_deg = np.degrees(q_model[:6])
        arm_deg_str = ", ".join(f"{v:+.2f}°" for v in arm_deg)
        arm_rad_str = ", ".join(f"{v:+.4f}" for v in q_model[:6])

        # Cache the current pose data for saving on spacebar
        self.latest_pose_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "ee_frame": self.ee_frame,
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "rpy_deg": [float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])],
            "rpy_rad": [float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2])],
            "quaternion": [float(quat.x), float(quat.y), float(quat.z), float(quat.w)],
            "joints_deg": [float(v) for v in arm_deg],
            "joints_rad": [float(v) for v in q_model[:6]],
            "arm_deg_str": arm_deg_str,
            "arm_rad_str": arm_rad_str,
        }

        # Check print throttle interval
        if now - self.last_print_time < self.print_interval:
            return

        # Check movement threshold if change_only is enabled
        if self.change_only and self.last_pos is not None and self.last_rpy_deg is not None:
            pos_delta = float(np.linalg.norm(position - self.last_pos))
            rot_delta = float(np.max(np.abs(rpy_deg - self.last_rpy_deg)))
            if pos_delta < self.pos_threshold and rot_delta < self.rot_threshold:
                return

        self.last_print_time = now
        self.last_pos = position.copy()
        self.last_rpy_deg = rpy_deg.copy()

        if self.compact:
            # Compact 1-line overwrite mode
            msg_line = (
                f"\r[Pose] X:{position[0]:+.4f} Y:{position[1]:+.4f} Z:{position[2]:+.4f} m | "
                f"RPY(deg): R:{rpy_deg[0]:+.2f}° P:{rpy_deg[1]:+.2f}° Y:{rpy_deg[2]:+.2f}° | "
                f"RPY(rad): R:{rpy_rad[0]:+.4f} P:{rpy_rad[1]:+.4f} Y:{rpy_rad[2]:+.4f} | "
                f"[Saved: {self.waypoint_count}]"
            )
            sys.stdout.write(msg_line)
            sys.stdout.flush()
            self.latest_pose_str = msg_line
        else:
            block = (
                f"\n{'='*72}\n"
                f"  reBotArm Pose Readout (Frame: {self.ee_frame}) | Saved Waypoints: {self.waypoint_count}\n"
                f"{'-'*72}\n"
                f"  Position (m)    : X = {position[0]:+.4f}, Y = {position[1]:+.4f}, Z = {position[2]:+.4f}\n"
                f"  Euler RPY (deg) : Roll = {rpy_deg[0]:+.2f}°, Pitch = {rpy_deg[1]:+.2f}°, Yaw = {rpy_deg[2]:+.2f}°\n"
                f"  Euler RPY (rad) : Roll = {rpy_rad[0]:+.4f}, Pitch = {rpy_rad[1]:+.4f}, Yaw = {rpy_rad[2]:+.4f}\n"
                f"  Quaternion      : [qx={quat.x:+.4f}, qy={quat.y:+.4f}, qz={quat.z:+.4f}, qw={quat.w:+.4f}]\n"
                f"  Joints (deg)    : [{arm_deg_str}]\n"
                f"  IK Target Format: x={position[0]:.4f}, y={position[1]:.4f}, z={position[2]:.4f}, "
                f"roll={rpy_rad[0]:.4f}, pitch={rpy_rad[1]:.4f}, yaw={rpy_rad[2]:.4f}\n"
                f"{'='*72}"
            )
            print(block)
            self.latest_pose_str = block

    def save_current_pose(self) -> bool:
        """Save the latest pose to plaintext and json files."""
        if self.latest_pose_data is None:
            if self.compact:
                sys.stdout.write("\n")
            print("[WARN] Cannot save: no joint states received yet. Move the arm slightly.")
            return False

        self.waypoint_count += 1
        data = self.latest_pose_data
        pos = data["position"]
        rpy_deg = data["rpy_deg"]
        rpy_rad = data["rpy_rad"]
        quat = data["quaternion"]
        timestamp = data["timestamp"]

        # 1. Append formatted human-readable block to plaintext file
        entry_txt = (
            f"================================================================================\n"
            f"[Waypoint #{self.waypoint_count}] Timestamp: {timestamp} | Frame: {data['ee_frame']}\n"
            f"--------------------------------------------------------------------------------\n"
            f"  Position (m)    : X = {pos[0]:+.4f}, Y = {pos[1]:+.4f}, Z = {pos[2]:+.4f}\n"
            f"  Euler RPY (deg) : Roll = {rpy_deg[0]:+.2f}°, Pitch = {rpy_deg[1]:+.2f}°, Yaw = {rpy_deg[2]:+.2f}°\n"
            f"  Euler RPY (rad) : Roll = {rpy_rad[0]:+.4f}, Pitch = {rpy_rad[1]:+.4f}, Yaw = {rpy_rad[2]:+.4f}\n"
            f"  Quaternion      : [qx={quat[0]:+.4f}, qy={quat[1]:+.4f}, qz={quat[2]:+.4f}, qw={quat[3]:+.4f}]\n"
            f"  Joints (deg)    : [{data['arm_deg_str']}]\n"
            f"  Joints (rad)    : [{data['arm_rad_str']}]\n"
            f"  IK Target Format: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}, "
            f"roll={rpy_rad[0]:.4f}, pitch={rpy_rad[1]:.4f}, yaw={rpy_rad[2]:.4f}\n"
            f"  Python Dict:\n"
            f"    {{\"x\": {pos[0]:.4f}, \"y\": {pos[1]:.4f}, \"z\": {pos[2]:.4f}, "
            f"\"roll\": {rpy_rad[0]:.4f}, \"pitch\": {rpy_rad[1]:.4f}, \"yaw\": {rpy_rad[2]:.4f}}}\n"
            f"================================================================================\n\n"
        )

        with open(self.save_file_txt, "a", encoding="utf-8") as f:
            f.write(entry_txt)

        # 2. Update JSON file for programmatic reference
        wp_record = {
            "waypoint_id": self.waypoint_count,
            "timestamp": timestamp,
            "ee_frame": data["ee_frame"],
            "position": pos,
            "rpy_deg": rpy_deg,
            "rpy_rad": rpy_rad,
            "quaternion": quat,
            "joints_deg": data["joints_deg"],
            "joints_rad": data["joints_rad"],
            "ik_target": {
                "x": round(pos[0], 4),
                "y": round(pos[1], 4),
                "z": round(pos[2], 4),
                "roll": round(rpy_rad[0], 4),
                "pitch": round(rpy_rad[1], 4),
                "yaw": round(rpy_rad[2], 4),
            },
        }
        self.saved_waypoints.append(wp_record)
        with open(self.save_file_json, "w", encoding="utf-8") as f:
            json.dump(self.saved_waypoints, f, indent=2)

        # 3. Print clear visual notification on terminal
        if self.compact:
            sys.stdout.write("\n")
        print(
            f"\n>>> [SAVED WAYPOINT #{self.waypoint_count}] "
            f"X={pos[0]:+.4f}, Y={pos[1]:+.4f}, Z={pos[2]:+.4f} | "
            f"RPY={rpy_deg[0]:+.1f}°, {rpy_deg[1]:+.1f}°, {rpy_deg[2]:+.1f}°\n"
            f"    File: {self.save_file_txt}\n"
        )
        return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gravity compensation with real-time XYZ and RPY pose readout and waypoint saving."
    )
    parser.add_argument(
        "--namespace",
        default="rebotarm",
        help="ROS namespace for the arm (default: 'rebotarm')",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="Pose print rate in Hz (default: 5.0)",
    )
    parser.add_argument(
        "--ee-frame",
        type=str,
        default="",
        help="End-effector frame name override (defaults to SDK config value)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact single-line updates using carriage return",
    )
    parser.add_argument(
        "--change-only",
        action="store_true",
        help="Only print when arm movement exceeds thresholds",
    )
    parser.add_argument(
        "--pos-threshold",
        type=float,
        default=0.002,
        help="Position change threshold in meters for --change-only (default: 0.002 m)",
    )
    parser.add_argument(
        "--rot-threshold",
        type=float,
        default=0.5,
        help="Orientation change threshold in degrees for --change-only (default: 0.5 deg)",
    )
    parser.add_argument(
        "--no-safe-home",
        action="store_true",
        help="Skip safe_home on exit to keep arm in current pose",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="saved_poses",
        help="Folder to save recorded pose readouts (default: 'saved_poses')",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default="",
        help="Custom filename for saving poses (default: 'poses_<timestamp>.txt')",
    )
    clean_argv = rclpy.utilities.remove_ros_args(sys.argv[1:])
    return parser.parse_args(args=clean_argv)


def main() -> None:
    args = _parse_args()
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = GravityCompensationPoseNode(args)
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        if not stop_requested:
            node.get_logger().info("stop requested, shutting down gravity compensation")
        stop_requested = True

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    # Prepare non-blocking terminal input for detecting Spacebar
    is_tty = sys.stdin.isatty()
    old_term_settings = None
    if is_tty:
        try:
            fd = sys.stdin.fileno()
            old_term_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception as exc:
            node.get_logger().warn(f"Failed to set terminal to cbreak mode: {exc}")
            is_tty = False

    ns = node.namespace
    enable_client = node.create_client(Trigger, f"/{ns}/enable")
    start_client = node.create_client(Trigger, f"/{ns}/gravity_compensation/start")
    safe_home_client = node.create_client(Trigger, f"/{ns}/safe_home")
    disable_client = node.create_client(Trigger, f"/{ns}/disable")

    gc_started = False
    try:
        node.get_logger().info(f"Connecting to /{ns} services...")
        if not _call_trigger(node, enable_client, "enable"):
            raise SystemExit(1)
        if not _call_trigger(node, start_client, "start gravity compensation"):
            raise SystemExit(1)
        gc_started = True

        print("\n" + "#" * 72)
        print("  Gravity Compensation ACTIVE — You can now manually move the arm.")
        print(f"  Streaming End-Effector XYZ and RPY at {args.rate:.1f} Hz")
        print(f"  Poses will be saved to: {node.save_file_txt}")
        print("  Controls:")
        print("    [SPACE]   Save current pose / waypoint to file")
        print("    [Ctrl+C]  Stop gravity compensation and safely exit")
        print("#" * 72 + "\n")

        while rclpy.ok() and not stop_requested:
            rclpy.spin_once(node, timeout_sec=0.04)

            # Check for keypresses (Spacebar or quit)
            if is_tty:
                r, _, _ = select.select([sys.stdin], [], [], 0.0)
                if r:
                    ch = sys.stdin.read(1)
                    if ch == " ":
                        node.save_current_pose()
                    elif ch in ("\x03", "q", "Q"):
                        stop_requested = True

    finally:
        # Restore terminal settings
        if is_tty and old_term_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_term_settings)
            except Exception:
                pass

        if args.compact:
            sys.stdout.write("\n")
            sys.stdout.flush()

        if node.waypoint_count > 0:
            print(
                f"\n[DONE] Recorded {node.waypoint_count} waypoint(s) into:\n"
                f"  Text: {node.save_file_txt}\n"
                f"  JSON: {node.save_file_json}\n"
            )

        if gc_started:
            if not args.no_safe_home:
                try:
                    node.get_logger().info("Homing arm safely before shutdown...")
                    _call_trigger(node, safe_home_client, "safe_home", timeout_sec=35.0)
                except Exception as exc:
                    node.get_logger().warn(f"safe_home cleanup failed: {exc}")
            else:
                node.get_logger().info("Skipping safe_home (--no-safe-home enabled)")

            try:
                _call_trigger(node, disable_client, "disable")
            except Exception as exc:
                node.get_logger().warn(f"disable cleanup failed: {exc}")

        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

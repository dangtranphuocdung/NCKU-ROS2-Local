#!/usr/bin/env python3
from __future__ import annotations

import time
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, Int8
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    from rebotarm_msgs.srv import GripperCommand as RebotGripperCommandSrv
except ImportError:
    RebotGripperCommandSrv = None


class GraspManagerNode(Node):
    """
    Grasp Manager State Machine for reBotArm.
    Uses ONLY vendor-provided ROS APIs to command the physical arm:
      - Arm motion: /{arm_namespace}/follow_joint_trajectory (Action)
      - Gripper motion: /{arm_namespace}/gripper/command (Action) / /{arm_namespace}/gripper/open/close (Srv)
      - Hardware enable: /{arm_namespace}/enable (Trigger)
      - Feedback: /{arm_namespace}/joint_states & /joint_states
      - IBVS Servoing: MoveIt Servo (/servo_node) streamed via hardware_servo_bridge_node to /{arm_namespace}/joints/*/cmd/pos_vel
    """

    def __init__(self) -> None:
        super().__init__('grasp_manager_node')

        # ----------------------------------------------------------------------
        # 1. PARAMETERS
        # ----------------------------------------------------------------------
        self.declare_parameter('arm_namespace', 'rebotarm')
        self.declare_parameter('joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('gripper_joint_names', ['gripper_joint1', 'gripper_joint2'])

        # Calibrated joint scanning pose and timings
        self.declare_parameter('scanning_joints', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('scanning_pose_duration', 3.0)
        self.declare_parameter('joint_tolerance', 0.03)

        # Gripper parameters
        self.declare_parameter('open_gripper_position', 0.045)
        self.declare_parameter('grasp_gripper_position', 0.017)
        self.declare_parameter('hardware_gripper_action_name', '/rebotarm/gripper/command')
        self.declare_parameter('hardware_open_gripper_position', -3.0)
        self.declare_parameter('hardware_closed_gripper_position', 0.0)
        self.declare_parameter('max_gripper_width', 0.09)
        self.declare_parameter('gripper_max_effort', 10.0)
        self.declare_parameter('gripper_motion_duration', 1.5)

        # Visual Servoing / YOLO parameters
        self.declare_parameter('target_pixel_x', 320.0)
        self.declare_parameter('target_pixel_y', 240.0)
        self.declare_parameter('target_area_px', 40000.0)
        self.declare_parameter('tolerance_px', 30.0)
        self.declare_parameter('tolerance_area', 5000.0)
        self.declare_parameter('convergence_hold_time', 0.8)
        self.declare_parameter('tracking_timeout', 25.0)

        # Extract values
        self.arm_namespace = str(self.get_parameter('arm_namespace').value).strip('/')
        self.joint_names = [str(n) for n in self.get_parameter('joint_names').value]
        self.gripper_joint_names = [str(n) for n in self.get_parameter('gripper_joint_names').value]

        self.scanning_joints = [float(j) for j in self.get_parameter('scanning_joints').value]
        self.scanning_pose_duration = float(self.get_parameter('scanning_pose_duration').value)
        self.joint_tolerance = float(self.get_parameter('joint_tolerance').value)

        self.open_gripper_pos = float(self.get_parameter('open_gripper_position').value)
        self.grasp_gripper_pos = float(self.get_parameter('grasp_gripper_position').value)
        self.hw_gripper_action_name = str(self.get_parameter('hardware_gripper_action_name').value)
        self.hw_open_pos = float(self.get_parameter('hardware_open_gripper_position').value)
        self.hw_closed_pos = float(self.get_parameter('hardware_closed_gripper_position').value)
        self.max_gripper_width = float(self.get_parameter('max_gripper_width').value)
        self.gripper_max_effort = float(self.get_parameter('gripper_max_effort').value)
        self.gripper_duration = float(self.get_parameter('gripper_motion_duration').value)

        self.target_pixel_x = float(self.get_parameter('target_pixel_x').value)
        self.target_pixel_y = float(self.get_parameter('target_pixel_y').value)
        self.target_area_px = float(self.get_parameter('target_area_px').value)
        self.tolerance_px = float(self.get_parameter('tolerance_px').value)
        self.tolerance_area = float(self.get_parameter('tolerance_area').value)
        self.hold_time = float(self.get_parameter('convergence_hold_time').value)
        self.tracking_timeout = float(self.get_parameter('tracking_timeout').value)

        # ----------------------------------------------------------------------
        # 2. VENDOR API CLIENTS & SUBSCRIBERS
        # ----------------------------------------------------------------------
        # 2.1 Arm trajectory action (Vendor primary API)
        hw_traj_topic = f'/{self.arm_namespace}/follow_joint_trajectory'
        self._arm_traj_client = ActionClient(self, FollowJointTrajectory, hw_traj_topic)
        self._sim_traj_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')

        # 2.2 Gripper action clients (Hardware and simulation)
        self._gripper_action_client = ActionClient(self, GripperCommand, self.hw_gripper_action_name)
        self._sim_gripper_client = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')

        # 2.3 Vendor direct hardware services
        self._enable_client = self.create_client(Trigger, f'/{self.arm_namespace}/enable')
        if RebotGripperCommandSrv is not None:
            self._hw_gripper_open_srv = self.create_client(RebotGripperCommandSrv, f'/{self.arm_namespace}/gripper/open')
            self._hw_gripper_close_srv = self.create_client(RebotGripperCommandSrv, f'/{self.arm_namespace}/gripper/close')
        else:
            self._hw_gripper_open_srv = None
            self._hw_gripper_close_srv = None

        # 2.4 MoveIt Servo Control & Status (used during visual servoing phase)
        self.servo_start_cli = self.create_client(Trigger, '/servo_node/start_servo')
        self.servo_stop_cli = self.create_client(Trigger, '/servo_node/stop_servo')
        self.servo_unpause_cli = self.create_client(Trigger, '/servo_node/unpause_servo')
        self.create_subscription(Int8, '/servo_node/status', self._servo_status_cb, 10)

        # 2.5 IBVS Vision & Controller Interface
        self.ibvs_enable_pub = self.create_publisher(Bool, '/ibvs/enable', 10)
        self.create_subscription(Float32MultiArray, '/yolo/target_bounding_box', self._bbox_cb, 10)

        # 2.6 Joint state tracking
        self._current_joint_positions: dict[str, float] = {}
        self.create_subscription(JointState, f'/{self.arm_namespace}/joint_states', self._joint_state_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, qos_profile_sensor_data)

        # ----------------------------------------------------------------------
        # 3. STATE MACHINE VARIABLES
        # ----------------------------------------------------------------------
        self.state = 'STARTUP'
        self.busy = False
        self.current_error_x = 999.0
        self.current_error_y = 999.0
        self.current_error_area = 99999.0
        self.convergence_start_time: float | None = None
        self.tracking_start_time: float | None = None
        self._active_timers: list[rclpy.timer.Timer] = []

        # 10 Hz periodic state machine loop
        self.timer = self.create_timer(0.1, self.state_machine_loop)
        self.get_logger().info(
            f"GraspManagerNode initialized using native vendor API /{self.arm_namespace}/follow_joint_trajectory."
        )

    # --------------------------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------------------------
    def _joint_state_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self._current_joint_positions[name] = float(pos)

    def _bbox_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 4:
            cx, cy, w, h = msg.data
            area = w * h
            self.current_error_x = abs(self.target_pixel_x - cx)
            self.current_error_y = abs(self.target_pixel_y - cy)
            self.current_error_area = abs(self.target_area_px - area)

    def _servo_status_cb(self, msg: Int8) -> None:
        status_map = {
            0: "NO_WARNING (Ready & Active)",
            1: "DECELERATING_FOR_APPROACHING_SINGULARITY",
            2: "HALT_FOR_SINGULARITY",
            3: "DECELERATING_FOR_COLLISION",
            4: "HALT_FOR_COLLISION",
            5: "JOINT_BOUND",
        }
        name = status_map.get(msg.data, f"CODE_{msg.data}")
        if msg.data != 0:
            self.get_logger().warn(
                f"[MoveIt Servo Status Alert]: {name}",
                throttle_duration_sec=2.0,
            )
        else:
            self.get_logger().info(
                f"[MoveIt Servo Status]: {name}",
                throttle_duration_sec=10.0,
            )

    def set_ibvs(self, enable: bool) -> None:
        msg = Bool()
        msg.data = enable
        self.ibvs_enable_pub.publish(msg)

    @staticmethod
    def duration(seconds: float) -> Duration:
        sec = int(seconds)
        nanosec = int((float(seconds) - sec) * 1e9)
        return Duration(sec=sec, nanosec=nanosec)

    def _oneshot_timer(self, delay: float, callback) -> None:
        timer_ref = None

        def _wrapper():
            nonlocal timer_ref
            if timer_ref is not None:
                timer_ref.cancel()
                self.destroy_timer(timer_ref)
                if timer_ref in self._active_timers:
                    self._active_timers.remove(timer_ref)
            callback()

        timer_ref = self.create_timer(delay, _wrapper)
        self._active_timers.append(timer_ref)

    # --------------------------------------------------------------------------
    # VENDOR ARM MOTION: FollowJointTrajectory Action
    # --------------------------------------------------------------------------
    def move_arm_to_joints(
        self,
        goal_values: list[float],
        duration: float,
        label: str,
        on_success_callback,
    ) -> None:
        """
        Commands arm to joint targets directly using vendor FollowJointTrajectory action server.
        Matches vendor examples/move_to.py.
        """
        current_values = [self._current_joint_positions.get(name, 0.0) for name in self.joint_names]
        self.get_logger().info(
            f"Commanding arm to {label} ({duration:.1f}s): "
            f"target={[round(v, 3) for v in goal_values]}, "
            f"current={[round(v, 3) for v in current_values]}"
        )

        if all(abs(c - g) <= self.joint_tolerance for c, g in zip(current_values, goal_values)):
            self.get_logger().info(f"Arm already at {label} (within {self.joint_tolerance:.3f} rad). Proceeding.")
            on_success_callback()
            return

        # Select available trajectory server
        client = None
        if self._arm_traj_client.server_is_ready():
            client = self._arm_traj_client
        elif self._sim_traj_client.server_is_ready():
            client = self._sim_traj_client
        else:
            self.get_logger().info("Waiting for follow_joint_trajectory action server...")
            if self._arm_traj_client.wait_for_server(timeout_sec=3.0):
                client = self._arm_traj_client
            elif self._sim_traj_client.wait_for_server(timeout_sec=1.0):
                client = self._sim_traj_client

        if client is None:
            self.get_logger().error(
                f"Trajectory server /{self.arm_namespace}/follow_joint_trajectory not available! Cannot move arm."
            )
            self.busy = False
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        # Must match self._hardware.joint_names in vendor controller
        goal.trajectory.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in goal_values]
        point.time_from_start = self.duration(duration)
        goal.trajectory.points = [point]

        self.get_logger().info(f"Sending FollowJointTrajectory goal for {label} to controller...")
        send_future = client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda f: self._on_traj_goal_sent(f, label, on_success_callback)
        )

    def _on_traj_goal_sent(self, future, label: str, on_success_callback) -> None:
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error(f"FollowJointTrajectory goal for {label} REJECTED by controller.")
            self.busy = False
            return

        self.get_logger().info(f"FollowJointTrajectory goal for {label} ACCEPTED. Executing motion...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._on_traj_result_received(f, label, on_success_callback)
        )

    def _on_traj_result_received(self, future, label: str, on_success_callback) -> None:
        action_res = future.result()
        if action_res and action_res.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info(f"Motion to {label} COMPLETED successfully on hardware!")
        else:
            code = action_res.result.error_code if action_res else "None"
            msg = action_res.result.error_string if action_res else ""
            self.get_logger().warn(f"FollowJointTrajectory for {label} finished with code={code}: {msg}")

        on_success_callback()

    # --------------------------------------------------------------------------
    # VENDOR GRIPPER CONTROL: Action & Service APIs
    # --------------------------------------------------------------------------
    def open_gripper(self, on_success_callback) -> None:
        """Opens gripper using vendor service /{arm_namespace}/gripper/open (fastest) or action."""
        if self._hw_gripper_open_srv is not None and (
            self._hw_gripper_open_srv.service_is_ready() or self._hw_gripper_open_srv.wait_for_service(timeout_sec=0.2)
        ):
            self.get_logger().info(f"[Vendor Service] Calling /{self.arm_namespace}/gripper/open...")
            req = RebotGripperCommandSrv.Request()
            req.position = 0.0  # 0.0 uses controller default calibrated open position
            req.timeout = 3.0
            future = self._hw_gripper_open_srv.call_async(req)
            future.add_done_callback(lambda f: self._on_gripper_srv_done(f, "open", on_success_callback))
            return

        self.command_gripper(self.open_gripper_pos, on_success_callback)

    def close_gripper(self, on_success_callback) -> None:
        """Closes/grasps gripper using vendor service /{arm_namespace}/gripper/close (fastest) or action."""
        if self._hw_gripper_close_srv is not None and (
            self._hw_gripper_close_srv.service_is_ready() or self._hw_gripper_close_srv.wait_for_service(timeout_sec=0.2)
        ):
            self.get_logger().info(f"[Vendor Service] Calling /{self.arm_namespace}/gripper/close...")
            req = RebotGripperCommandSrv.Request()
            req.position = 0.0  # 0.0 uses controller default calibrated close position
            req.timeout = 3.0
            future = self._hw_gripper_close_srv.call_async(req)
            future.add_done_callback(lambda f: self._on_gripper_srv_done(f, "close", on_success_callback))
            return

        self.command_gripper(self.grasp_gripper_pos, on_success_callback)

    def _on_gripper_srv_done(self, future, action_name: str, on_success_callback) -> None:
        try:
            res = future.result()
            self.get_logger().info(
                f"[Vendor Service] Gripper {action_name} complete: success={res.success}, "
                f"reached={res.reached_position:.3f} rad, msg={res.message}"
            )
        except Exception as e:
            self.get_logger().warn(f"Gripper {action_name} service call failed: {e}")
        on_success_callback()

    def command_gripper(self, target_position_m: float, on_success_callback) -> None:
        """
        Commands gripper via vendor /{arm_namespace}/gripper/command action or services.
        """
        self.get_logger().info(f"Commanding gripper (metric target: {target_position_m:.4f} m)...")

        # 1. Hardware GripperCommand action
        if self._gripper_action_client.server_is_ready() or self._gripper_action_client.wait_for_server(timeout_sec=0.5):
            ratio = 0.0 if self.max_gripper_width <= 0.0 else (2.0 * target_position_m / self.max_gripper_width)
            ratio = max(0.0, min(1.0, ratio))
            hw_target = self.hw_closed_pos + (self.hw_open_pos - self.hw_closed_pos) * ratio

            goal = GripperCommand.Goal()
            goal.command.position = hw_target
            goal.command.max_effort = self.gripper_max_effort

            self.get_logger().info(
                f"[Hardware Gripper] Sending GripperCommand to {self.hw_gripper_action_name}: "
                f"pos={hw_target:.3f} rad, effort={self.gripper_max_effort}"
            )
            future = self._gripper_action_client.send_goal_async(goal)
            future.add_done_callback(lambda f: self._on_gripper_action_sent(f, on_success_callback))
            return

        # 2. Simulation FollowJointTrajectory action
        if self._sim_gripper_client.server_is_ready() or self._sim_gripper_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().info(f"[Simulation Gripper] Sending FollowJointTrajectory goal: {target_position_m:.4f} m")
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = JointTrajectory(
                joint_names=list(self.gripper_joint_names),
                points=[
                    JointTrajectoryPoint(
                        positions=[target_position_m] * len(self.gripper_joint_names),
                        time_from_start=self.duration(self.gripper_duration),
                    )
                ],
            )
            future = self._sim_gripper_client.send_goal_async(goal)
            future.add_done_callback(lambda f: self._on_gripper_action_sent(f, on_success_callback))
            return

        # 3. Fallback: wait duration and proceed
        self.get_logger().warn("Gripper action server not found; proceeding after motion delay.")
        self._oneshot_timer(self.gripper_duration, on_success_callback)

    def _on_gripper_action_sent(self, future, on_success_callback) -> None:
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn("Gripper goal rejected or not accepted; continuing.")
            on_success_callback()
            return

        res_future = goal_handle.get_result_async()
        res_future.add_done_callback(lambda f: on_success_callback())

    # --------------------------------------------------------------------------
    # STATE MACHINE LOOP (10 Hz)
    # --------------------------------------------------------------------------
    def state_machine_loop(self) -> None:
        if self.busy:
            return

        # ----------------------------------------------------------------------
        # STATE 1: STARTUP -> Ensure Hardware Connected & Enabled, Open Gripper
        # ----------------------------------------------------------------------
        if self.state == 'STARTUP':
            if not all(name in self._current_joint_positions for name in self.joint_names):
                self.get_logger().info(
                    f"Waiting for joint states from /{self.arm_namespace}/joint_states...",
                    throttle_duration_sec=2.0,
                )
                return

            self.busy = True
            current_vals = [round(self._current_joint_positions[n], 3) for n in self.joint_names]
            self.get_logger().info(f"--- Hardware connected! Current joint positions: {current_vals} ---")

            # Enable hardware actuators via vendor service
            if self._enable_client.service_is_ready():
                self.get_logger().info(f"Calling /{self.arm_namespace}/enable...")
                self._enable_client.call_async(Trigger.Request())

            self.get_logger().info("--- Phase 1: Opening Gripper ---")
            self.open_gripper(self._on_initial_gripper_opened)

        # ----------------------------------------------------------------------
        # STATE 2: ENABLE_IBVS -> Start Servo & Visual Tracking
        # ----------------------------------------------------------------------
        elif self.state == 'ENABLE_IBVS':
            self.busy = True
            self.get_logger().info("--- Phase 2: Starting MoveIt Servo & IBVS Tracking ---")
            self._start_servo_and_tracking()

        # ----------------------------------------------------------------------
        # STATE 3: WAIT_CONVERGENCE -> Visual Servoing Tracking Loop
        # ----------------------------------------------------------------------
        elif self.state == 'WAIT_CONVERGENCE':
            self.set_ibvs(True)
            now = time.time()

            # Safety Timeout
            if self.tracking_start_time and (now - self.tracking_start_time > self.tracking_timeout):
                self.get_logger().warn(
                    f"Tracking timeout ({self.tracking_timeout:.1f}s) reached. Halting Servo and proceeding to grasp..."
                )
                self.set_ibvs(False)
                if self.servo_stop_cli.service_is_ready():
                    self.servo_stop_cli.call_async(Trigger.Request())
                self.state = 'GRASP'
                return

            # Check alignment convergence
            has_valid_detection = (self.current_error_x < 900.0 and self.current_error_y < 900.0)
            within_tolerance = (
                self.current_error_x < self.tolerance_px
                and self.current_error_y < self.tolerance_px
                and self.current_error_area < self.tolerance_area
            )

            if has_valid_detection and within_tolerance:
                if self.convergence_start_time is None:
                    self.convergence_start_time = now
                elif now - self.convergence_start_time >= self.hold_time:
                    self.get_logger().info("IBVS Converged! Object aligned. Halting Servo...")
                    self.set_ibvs(False)
                    if self.servo_stop_cli.service_is_ready():
                        self.servo_stop_cli.call_async(Trigger.Request())
                    self.state = 'GRASP'
            else:
                self.convergence_start_time = None
                if has_valid_detection:
                    self.get_logger().info(
                        f"Tracking target... Errors: X={self.current_error_x:.1f}px, "
                        f"Y={self.current_error_y:.1f}px, Area={self.current_error_area:.1f}px^2",
                        throttle_duration_sec=1.0,
                    )
                else:
                    self.get_logger().warn(
                        "Waiting for YOLO detection in camera view...",
                        throttle_duration_sec=2.0,
                    )

        # ----------------------------------------------------------------------
        # STATE 4: GRASP -> Close Gripper
        # ----------------------------------------------------------------------
        elif self.state == 'GRASP':
            self.busy = True
            self.get_logger().info("--- Phase 3: Grasping Object ---")
            self.close_gripper(self._on_grasp_complete)

        # ----------------------------------------------------------------------
        # STATE 5: RETREAT -> Lift Back to Scanning Pose
        # ----------------------------------------------------------------------
        elif self.state == 'RETREAT':
            self.busy = True
            self.get_logger().info("--- Phase 4: Retreating to Scanning Pose via vendor API ---")
            self.move_arm_to_joints(
                self.scanning_joints,
                self.scanning_pose_duration,
                "retreat_scanning_pose",
                self._on_retreat_complete,
            )

        # ----------------------------------------------------------------------
        # STATE 6: DONE
        # ----------------------------------------------------------------------
        elif self.state == 'DONE':
            pass

    # --------------------------------------------------------------------------
    # STATE TRANSITIONS
    # --------------------------------------------------------------------------
    def _on_initial_gripper_opened(self) -> None:
        self.get_logger().info("Gripper opened. Moving arm to scanning pose via vendor follow_joint_trajectory...")
        self.move_arm_to_joints(
            self.scanning_joints,
            self.scanning_pose_duration,
            "scanning_pose",
            self._on_scanning_pose_reached,
        )

    def _on_scanning_pose_reached(self) -> None:
        self.get_logger().info("Arrived at scanning pose! Transitioning to visual servoing.")
        self.busy = False
        self.state = 'ENABLE_IBVS'

    def _start_servo_and_tracking(self) -> None:
        self.get_logger().info("Activating MoveIt Servo and IBVS tracking...")
        if not self.servo_start_cli.service_is_ready():
            self.servo_start_cli.wait_for_service(timeout_sec=2.0)
        if self.servo_start_cli.service_is_ready():
            self.servo_start_cli.call_async(Trigger.Request())
            self.get_logger().info("Triggered /servo_node/start_servo")

        if not self.servo_unpause_cli.service_is_ready():
            self.servo_unpause_cli.wait_for_service(timeout_sec=1.0)
        if self.servo_unpause_cli.service_is_ready():
            self.servo_unpause_cli.call_async(Trigger.Request())
            self.get_logger().info("Triggered /servo_node/unpause_servo")

        self.set_ibvs(True)
        self.tracking_start_time = time.time()
        self.busy = False
        self.state = 'WAIT_CONVERGENCE'
        self.get_logger().info("MoveIt Servo & IBVS tracking active. State -> WAIT_CONVERGENCE")

    def _on_grasp_complete(self) -> None:
        self.get_logger().info("Object grasped! Transitioning to retreat.")
        self.busy = False
        self.state = 'RETREAT'

    def _on_retreat_complete(self) -> None:
        self.get_logger().info("======================================================")
        self.get_logger().info("  Visual Servoing Grasp Pipeline Succeeded! Done.   ")
        self.get_logger().info("======================================================")
        self.busy = False
        self.state = 'DONE'


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

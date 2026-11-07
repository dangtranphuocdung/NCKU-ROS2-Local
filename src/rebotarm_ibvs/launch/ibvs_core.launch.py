import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    # ── Launch arguments ──────────────────────────────────────────────────────
    use_hardware_arg = DeclareLaunchArgument(
        'use_hardware',
        default_value='true',
        description='Set false for sim-only (skips hardware_servo_bridge_node)',
    )
    arm_namespace_arg = DeclareLaunchArgument(
        'arm_namespace',
        default_value='rebotarm',
        description='Arm namespace (must match bringup.launch.py arm_namespace)',
    )

    # ── MoveIt config (mirrors hardware.launch.py) ────────────────────────────
    moveit_config = (
        MoveItConfigsBuilder('rebotarm', package_name='rebotarm_moveit_config')
        .robot_description(file_path='config/rebotarm.urdf.xacro')
        .robot_description_semantic(file_path='config/rebotarm.srdf')
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .joint_limits(file_path='config/joint_limits.yaml')
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )

    # ── Servo config ──────────────────────────────────────────────────────────
    servo_yaml_path = os.path.join(
        get_package_share_directory('rebotarm_ibvs'), 'config', 'servo.yaml'
    )
    with open(servo_yaml_path, 'r') as f:
        servo_params = yaml.safe_load(f)

    # ── Pipeline config (shared by ibvs_controller + hardware_bridge) ────────
    pipeline_config = os.path.expanduser('~/rebotarm_ros2/config/ibvs_pipeline.yaml')

    # ── Nodes ─────────────────────────────────────────────────────────────────

    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        name='servo_node',
        parameters=[
            {'moveit_servo': servo_params},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        output='screen',
    )

    # Eye-in-hand calibration: link6 → camera_link
    hand_eye_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='hand_eye_tf_publisher',
        # Values from config/calibration/hand_eye.npz
        # Format: x y z qx qy qz qw parent child
        arguments=[
            '-0.12848879', '0.00707998', '0.09285028',
            '0.60757216', '-0.58596113', '0.40408812', '-0.35244632',
            'link6', 'camera_link',
        ],
    )

    ibvs_controller_node = Node(
        package='rebotarm_ibvs',
        executable='ibvs_controller_node',
        name='ibvs_controller_node',
        parameters=[pipeline_config],
        output='screen',
    )

    # Hardware bridge — only when running on physical robot
    hardware_bridge_node = Node(
        package='rebotarm_ibvs',
        executable='hardware_servo_bridge_node',
        name='hardware_servo_bridge_node',
        parameters=[
            pipeline_config,
            {'arm_namespace': LaunchConfiguration('arm_namespace')},
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_hardware')),
    )

    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rebotarm_ibvs'),
                'launch', 'vision.launch.py',
            )
        )
    )

    return LaunchDescription([
        use_hardware_arg,
        arm_namespace_arg,
        hand_eye_tf_node,
        servo_node,
        ibvs_controller_node,
        hardware_bridge_node,
        vision_launch,
    ])

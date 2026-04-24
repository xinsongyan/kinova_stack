from __future__ import annotations

import os

from kinova_middleware.backend.config.robot_model import RobotModelConfig


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))

DEFAULT_MODEL_PATH = os.path.join(
    _ROOT_DIR, "kinova_description", "mjcf", "m1n4s300_standalone.mjcf"
)

DEFAULT_JOINT_ORDER = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_finger_1",
    "joint_finger_tip_1",
    "joint_finger_2",
    "joint_finger_tip_2",
    "joint_finger_3",
    "joint_finger_tip_3",
)

DEFAULT_FINGER_JOINTS = (
    "joint_finger_1",
    "joint_finger_tip_1",
    "joint_finger_2",
    "joint_finger_tip_2",
    "joint_finger_3",
    "joint_finger_tip_3",
)

DEFAULT_EE_BODY = "link_4"

DEFAULT_KINOVA_GEN3_LITE_CONFIG = RobotModelConfig(
    model_path=DEFAULT_MODEL_PATH,
    joint_names=DEFAULT_JOINT_ORDER,
    finger_joint_names=DEFAULT_FINGER_JOINTS,
    ee_body=DEFAULT_EE_BODY,
    ee_site=None,
    initial_keyframe="home",
    v_max_arm=(1.0, 1.0, 1.0, 1.0),
    a_max_arm=(2.0, 2.0, 2.0, 2.0),
    j_max_arm=(40.0, 40.0, 40.0, 40.0),
    omega_arm=(1.5, 2.0, 2.0, 2.0),
    kp_arm=(500.0, 300.0, 300.0, 300.0),
    kd_arm=(10.0, 30.0, 30.0, 30.0),
    kp_finger=5.0,
    kd_finger=0.005,
    finger_bias_mode="none",
    control_dt=0.001,
    j1_torque_limit=25.0,
)

DEFAULT_KINOVA_MUJOCO_CONFIG = DEFAULT_KINOVA_GEN3_LITE_CONFIG

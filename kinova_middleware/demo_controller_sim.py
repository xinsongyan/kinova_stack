#!/usr/bin/env python3
import math
import sys
import time

def euler_xyz_to_quaternion(theta_x: float, theta_y: float, theta_z: float) -> tuple[float, float, float, float]:
    """Convert XYZ intrinsic Euler angles (rad) to quaternion (qx, qy, qz, qw)."""
    cx = math.cos(theta_x * 0.5)
    sx = math.sin(theta_x * 0.5)
    cy = math.cos(theta_y * 0.5)
    sy = math.sin(theta_y * 0.5)
    cz = math.cos(theta_z * 0.5)
    sz = math.sin(theta_z * 0.5)

    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return (qx, qy, qz, qw)

from kinova_controller import KinovaController  # noqa: E402
from kinova_mujoco_backend import KinovaMuJoCoBackend  # noqa: E402

# ----------------------------
# User-configurable parameters
# ----------------------------
TARGET_X = None
TARGET_Y = None
TARGET_Z = None
TARGET_OFFSET = (0.05, 0.0, 0.0)

ROLL_DEG = 10.0
PITCH_DEG = -5.0
YAW_DEG = 20.0

JOINT_TARGET_DEG = [0.0, 88.0, 250.0, 0.0]

# Smooth target tracking to reduce jerk/jitter.
# Set to None to disable.
TARGET_SPEED_RAD_S = 1.5

# Low-pass filter for pose readings (set to 0.0 to disable; 0.1-0.3 is typical).
POSE_FILTER_ALPHA = 0.2


class PoseSmoother:
    def __init__(self, alpha: float) -> None:
        self._alpha = float(alpha)
        self._pos = None
        self._quat = None

    def update(self, pos: tuple[float, float, float], quat: tuple[float, float, float, float]):
        if self._alpha <= 0.0:
            return pos, quat
        if self._pos is None or self._quat is None:
            self._pos = pos
            self._quat = quat
            return pos, quat

        a = self._alpha
        self._pos = tuple((1.0 - a) * p + a * n for p, n in zip(self._pos, pos))
        self._quat = _quat_nlerp(self._quat, quat, a)
        return self._pos, self._quat


def _quat_nlerp(
    q_prev: tuple[float, float, float, float],
    q_new: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    dot = sum(p * n for p, n in zip(q_prev, q_new))
    if dot < 0.0:
        q_new = tuple(-v for v in q_new)
    blended = [(1.0 - alpha) * p + alpha * n for p, n in zip(q_prev, q_new)]
    norm = math.sqrt(sum(v * v for v in blended))
    if norm <= 1e-12:
        return q_prev
    return tuple(v / norm for v in blended)


def _run_steps(
    controller: KinovaController,
    seconds: float,
    hz: float = 200.0,
    hold_seconds: float = 2,
) -> None:
    dt = 1.0 / hz
    steps = max(1, int(seconds * hz))
    settled_since = None
    for _ in range(steps):
        reached = controller.step()
        if reached:
            if settled_since is None:
                settled_since = time.monotonic()
            if time.monotonic() - settled_since >= hold_seconds:
                break
        else:
            settled_since = None
        time.sleep(dt)

def main() -> int:
    print("Starting Kinova Controller...")
    backend = KinovaMuJoCoBackend(target_speed_rad_s=TARGET_SPEED_RAD_S)
    controller = KinovaController(backend)
    pose_smoother = PoseSmoother(POSE_FILTER_ALPHA)
    try:
        controller.init()
        print(f"Controller initialized (sim). DOF: {controller.dof}")

        print("Sending MoveHome command...")
        controller.move_home()
        _run_steps(controller, 2.0)

        print("Closing fingers...")
        controller.close_fingers(1.0)
        _run_steps(controller, 5.0)

        print("Opening fingers...")
        controller.open_fingers(1.0)
        _run_steps(controller, 5.0)

        print("Closing fingers...")
        controller.close_fingers(1.0)
        _run_steps(controller, 5.0)

        # for i in range(36):
        #     if len(JOINT_TARGET_DEG) < controller.arm_dof:
        #         raise ValueError(
        #             f"JOINT_TARGET_DEG expects at least {controller.arm_dof} values, "
        #             f"got {len(JOINT_TARGET_DEG)}."
        #         )
        #     joint_rad = [math.radians(v) for v in JOINT_TARGET_DEG[: controller.arm_dof]]
        #     print(f"Moving to joint targets (deg): {JOINT_TARGET_DEG[: controller.arm_dof]}")
        #     controller.send_joint_position_rad(joint_rad)
        #     _run_steps(controller, 10.0, hold_seconds=0.4)
        #     current_pos, current_quat = controller.get_end_effector_pose()
        #     current_pos, current_quat = pose_smoother.update(current_pos, current_quat)
        #     print(
        #         "Current pose:",
        #         f"x={current_pos[0]:.3f}, y={current_pos[1]:.3f}, z={current_pos[2]:.3f}",
        #         f"qx={current_quat[0]:.3f}, qy={current_quat[1]:.3f}, qz={current_quat[2]:.3f}, qw={current_quat[3]:.3f}"
        #     )
        #     time.sleep(1.0)
        #     JOINT_TARGET_DEG[0] += 10.0  # change first joint for next iteration


        pos, quat = controller.get_end_effector_pose()
        pos, quat = pose_smoother.update(pos, quat)
        print(f"EE pos: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
        print(f"EE quat: qx={quat[0]:.3f}, qy={quat[1]:.3f}, qz={quat[2]:.3f}, qw={quat[3]:.3f}")

        current_pos, current_quat = controller.get_end_effector_pose()
        current_pos, current_quat = pose_smoother.update(current_pos, current_quat)
        print(
            "Current pose:",
            f"x={current_pos[0]:.3f}, y={current_pos[1]:.3f}, z={current_pos[2]:.3f}",
            f"qx={current_quat[0]:.3f}, qy={current_quat[1]:.3f}, qz={current_quat[2]:.3f}, qw={current_quat[3]:.3f}"
        )

        target_pose = (-0.007, -0.000, 0.689)
        quat_target = (0.698, -0.716, -0.001, -0.001)
        print(
            "Target pose:",
            f"x={target_pose[0]:.3f}, y={target_pose[1]:.3f}, z={target_pose[2]:.3f}",
        )

        q_target = controller.plan_to_pose(target_pose, quat_target)
        _run_steps(controller, 2.0, target_joint_rad=q_target, hold_seconds=0.4)

        angles = controller.get_joint_angles_rad()
        print(f"Joint Angles: {angles}")

        velocities = controller.get_joint_vel_rad()
        print(f"Joint Velocities: {velocities}")

        inner_backend = getattr(controller._backend, "_inner", controller._backend)
        env = getattr(inner_backend, "_env", None)
        viewer = getattr(env, "viewer", None) if env is not None else None
        if viewer is not None:
            print("Close the MuJoCo viewer window to exit.")
            while True:
                controller.step()
                time.sleep(1.0 / 200.0)
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    print("Running Kinova Middleware Demo Controller (Simulated)...")
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(2)

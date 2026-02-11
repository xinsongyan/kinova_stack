import math
import unittest

import numpy as np

from kinova_backend import SafetyWrapperBackend
from kinova_controller import KinovaController
from kinova_mujoco_backend import KinovaMuJoCoBackend


POS_TOL = 0.03
ROT_TOL = 0.6


def quat_from_euler_xyz(theta_x: float, theta_y: float, theta_z: float) -> np.ndarray:
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
    return np.array([qx, qy, qz, qw], dtype=float)


def quat_wxyz_from_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_angle_error(q_target: np.ndarray, q_current: np.ndarray) -> float:
    q_err = quat_multiply(q_target, quat_conjugate(q_current))
    if q_err[0] < 0:
        q_err = -q_err
    q_err[0] = np.clip(q_err[0], -1.0, 1.0)
    return 2.0 * math.acos(float(q_err[0]))


class TestCartesianPoseSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        backend = KinovaMuJoCoBackend(viewer=False)
        safe_backend = SafetyWrapperBackend(
            backend,
            cartesian_bounds=([-0.6, -0.6, 0.0], [0.6, 0.6, 0.8]),
            on_violation="reject",
        )
        cls.controller = KinovaController(safe_backend, enforce_safety_wrapper=False)
        cls.controller.init()
        cls.backend = backend
        home_pos, home_quat = backend.get_end_effector_pose()
        cls.home_pos = np.array(home_pos, dtype=float)
        cls.home_quat = np.array(home_quat, dtype=float)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.controller.close()

    def _run_until(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        max_time: float = 2.0,
        hz: float = 200.0,
    ):
        dt = 1.0 / hz
        steps = max(1, int(max_time * hz))
        target_pos = np.array(target_pos, dtype=float)
        target_quat = quat_wxyz_from_xyzw(np.array(target_quat, dtype=float))

        pos_err = float("inf")
        rot_err = float("inf")
        for step_idx in range(steps):
            self.controller.step()
            pos, quat = self.backend.get_end_effector_pose()
            pos = np.array(pos, dtype=float)
            quat = quat_wxyz_from_xyzw(np.array(quat, dtype=float))
            pos_err = float(np.linalg.norm(target_pos - pos))
            rot_err = quat_angle_error(target_quat, quat)
            if pos_err <= POS_TOL and rot_err <= ROT_TOL:
                return pos_err, rot_err, (step_idx + 1) * dt
        return pos_err, rot_err, steps * dt

    def test_pose_reach(self) -> None:
        home_pos = self.home_pos
        home_quat = self.home_quat
        positions = [
            (home_pos[0] + 0.05, home_pos[1], home_pos[2]),
            (home_pos[0], home_pos[1] + 0.05, home_pos[2]),
            (home_pos[0], home_pos[1], home_pos[2] + 0.05),
            (home_pos[0] - 0.05, home_pos[1] - 0.05, home_pos[2]),
        ]

        for idx, pos in enumerate(positions, start=1):
            self.controller.plan_to_pose(pos, home_quat)
            pos_err, rot_err, ttc = self._run_until(pos, home_quat)
            ik_iters = self.backend.last_ik_iterations
            print(
                f"Pose {idx}: pos_err={pos_err:.4f} m, rot_err={rot_err:.3f} rad, "
                f"time_to_converge={ttc:.2f} s, ik_iters={ik_iters}"
            )
            self.assertLessEqual(pos_err, POS_TOL)
            self.assertLessEqual(rot_err, ROT_TOL)

    def test_orientation_only(self) -> None:
        home_pos = self.home_pos
        yaw_angles = [math.radians(10), math.radians(-15), math.radians(20)]
        for idx, yaw in enumerate(yaw_angles, start=1):
            q = quat_from_euler_xyz(0.0, 0.0, yaw)
            self.controller.plan_to_pose(home_pos, q)
            pos_err, rot_err, ttc = self._run_until(home_pos, q)
            ik_iters = self.backend.last_ik_iterations
            print(
                f"Orientation {idx}: pos_err={pos_err:.4f} m, rot_err={rot_err:.3f} rad, "
                f"time_to_converge={ttc:.2f} s, ik_iters={ik_iters}"
            )
            self.assertLessEqual(pos_err, POS_TOL)
            self.assertLessEqual(rot_err, ROT_TOL)

    def test_safety_rejects_out_of_bounds(self) -> None:
        home_pos = self.home_pos
        home_quat = self.home_quat
        out_pos = (home_pos[0] + 5.0, home_pos[1], home_pos[2])
        rejected = 0
        try:
            self.controller.plan_to_pose(out_pos, home_quat)
        except ValueError:
            rejected += 1
        self.assertEqual(rejected, 1)
        print(f"Rejected commands: {rejected}")


if __name__ == "__main__":
    unittest.main()

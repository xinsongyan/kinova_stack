from __future__ import annotations

import math
import time
from typing import Callable, Sequence

import numpy as np

from kinova_middleware.backend.runtime.mujoco_runtime import MuJoCoRuntimeAdapter
from kinova_sim.controller import ComputedTorqueController, PDController
from kinova_sim.governor import ReferenceGovernor
from kinova_sim.trajectory import TrajectoryGenerator


class MuJoCoArmControlService:
    """Own arm-target shaping, trajectory tracking, and arm torque control."""

    def __init__(
        self,
        runtime: MuJoCoRuntimeAdapter,
        arm_indices: Sequence[int],
        *,
        wrap_angle_fn: Callable[[float], float],
        v_max_arm: Sequence[float],
        a_max_arm: Sequence[float],
        j_max_arm: Sequence[float],
        omega_arm: Sequence[float],
        kp_arm: Sequence[float],
        kd_arm: Sequence[float],
        control_dt: float,
        j1_torque_limit: float,
    ) -> None:
        self._runtime = runtime
        self._arm_indices = tuple(int(i) for i in arm_indices)
        self._wrap_angle = wrap_angle_fn
        self._qpos_adr = runtime.qpos_adr
        self._qvel_adr = runtime.qvel_adr
        self._actuator_ids = runtime.actuator_ids
        self._v_max_arm = np.asarray(v_max_arm, dtype=float)
        self._a_max_arm = np.asarray(a_max_arm, dtype=float)
        self._j_max_arm = np.asarray(j_max_arm, dtype=float)
        self._omega_arm = np.asarray(omega_arm, dtype=float)
        self._kp_arm = np.asarray(kp_arm, dtype=float)
        self._kd_arm = np.asarray(kd_arm, dtype=float)
        self._control_dt = float(control_dt)
        self._j1_torque_limit = float(j1_torque_limit)

        physics_dt = float(runtime.model.opt.timestep)
        self._n_substeps = max(1, math.ceil(self._control_dt / physics_dt))
        effective_control_dt = self._n_substeps * physics_dt

        arm_dof_adrs = [self._qvel_adr[idx] for idx in self._arm_indices]
        q_init = np.array([float(runtime.data.qpos[adr]) for adr in self._qpos_adr], dtype=float)
        q_arm_init = np.array([q_init[i] for i in self._arm_indices], dtype=float)

        self._traj_gen = TrajectoryGenerator(
            n_joints=len(self._arm_indices),
            v_max=self._v_max_arm,
            a_max=self._a_max_arm,
            j_max=self._j_max_arm,
            omega=self._omega_arm,
            dt=effective_control_dt,
        )
        self._traj_gen.reset(q_arm_init)
        self._controller = ComputedTorqueController(
            Kp=self._kp_arm,
            Kd=self._kd_arm,
            arm_dof_adrs=arm_dof_adrs,
            nv=runtime.model.nv,
        )
        self._governor = ReferenceGovernor(torque_limit=self._j1_torque_limit)

    @property
    def n_substeps(self) -> int:
        return self._n_substeps

    def reset_from_joint_state(self, q_init_full: Sequence[float]) -> None:
        q_arm_init = np.array([float(q_init_full[i]) for i in self._arm_indices], dtype=float)
        self._traj_gen.reset(q_arm_init)

    def set_joint_target(
        self,
        q_des: Sequence[float],
        q_current_full: Sequence[float],
        q_target_desired: Sequence[float],
    ) -> np.ndarray:
        n_arm = len(self._arm_indices)
        is_arm_only = len(q_des) == n_arm

        q_full = np.array(q_target_desired, dtype=float).copy()
        q_des_wrapped = [float(v) for v in q_des]
        q_current = np.asarray(q_current_full, dtype=float)

        for idx in self._runtime.continuous_indices:
            if idx < len(q_des_wrapped):
                diff = q_des_wrapped[idx] - q_current[idx]
                q_des_wrapped[idx] -= 2 * np.pi * np.round(diff / (2 * np.pi))

        if is_arm_only:
            for i_local, idx_in_full in enumerate(self._arm_indices):
                q_full[idx_in_full] = q_des_wrapped[i_local]
            target_arm = np.array(q_des_wrapped, dtype=float)
        else:
            q_full = np.array(q_des_wrapped, dtype=float)
            target_arm = np.array([q_full[i] for i in self._arm_indices], dtype=float)

        q_full = self._clip_joint_targets(q_full)
        self._traj_gen.set_goal(target_arm)
        return q_full

    def compute_actuator_commands(self) -> list[tuple[int, float]]:
        model = self._runtime.model
        data = self._runtime.data

        q = np.array([data.qpos[adr] for adr in self._qpos_adr], dtype=float)
        qd = np.array([data.qvel[adr] for adr in self._qvel_adr], dtype=float)
        q_arm = np.array([q[i] for i in self._arm_indices], dtype=float)
        qd_arm = np.array([qd[i] for i in self._arm_indices], dtype=float)

        scale = self._governor.scale_current
        self._traj_gen.set_limits(
            0,
            v_max=self._v_max_arm[0] * scale,
            a_max=self._a_max_arm[0] * scale,
        )

        q_des, qd_des, qdd_des = self._traj_gen.update()
        if not (
            np.all(np.isfinite(q_des))
            and np.all(np.isfinite(qd_des))
            and np.all(np.isfinite(qdd_des))
        ):
            raise ValueError(
                "Trajectory generator emitted non-finite values: "
                f"q_des={q_des}, qd_des={qd_des}, qdd_des={qdd_des}"
            )
        if np.max(np.abs(qdd_des)) > 100.0:
            raise ValueError(
                "Trajectory generator emitted absurd acceleration limits "
                f"(max 100.0): qdd_des={qdd_des}"
            )

        tau_arm_raw, _, _ = self._controller.compute(
            model,
            data,
            q_des,
            qd_des,
            qdd_des,
            q_arm,
            qd_arm,
        )
        tau_raw_j1 = float(tau_arm_raw[0])
        self._governor.update(tau_raw_j1)

        commands: list[tuple[int, float]] = []
        for ai, arm_idx in enumerate(self._arm_indices):
            actuator_id = self._actuator_ids[arm_idx]
            tau_val = float(tau_arm_raw[ai])
            if ai == 0:
                limit = self._j1_torque_limit
                tau_cmd = float(limit * np.tanh(tau_val / limit)) if limit > 0 else tau_val
            else:
                tau_cmd = self._clip_actuator_force(actuator_id, tau_val)
            commands.append((actuator_id, tau_cmd))

        return commands

    def is_reached(
        self,
        q_target_desired: Sequence[float],
        pos_tol_rad: float,
        vel_tol_rad_s: float,
    ) -> bool:
        q_target = np.asarray(q_target_desired, dtype=float)
        q = np.array([self._runtime.data.qpos[adr] for adr in self._qpos_adr], dtype=float)
        qd = np.array([self._runtime.data.qvel[adr] for adr in self._qvel_adr], dtype=float)

        arm_pos_err = max(abs(q_target[i] - q[i]) for i in self._arm_indices)
        arm_vel = max(abs(qd[i]) for i in self._arm_indices)
        return bool(arm_pos_err <= float(pos_tol_rad) and arm_vel <= float(vel_tol_rad_s))

    def _clip_joint_targets(self, q: np.ndarray) -> np.ndarray:
        q_clipped = q.copy()
        for idx, (low, high) in enumerate(self._runtime.joint_limits):
            if np.isfinite(low) or np.isfinite(high):
                q_clipped[idx] = float(np.clip(q_clipped[idx], low, high))
        return q_clipped

    def _clip_actuator_force(self, actuator_id: int, value: float) -> float:
        model = self._runtime.model
        if model.actuator_forcelimited[actuator_id]:
            low, high = model.actuator_forcerange[actuator_id]
            return float(np.clip(value, low, high))
        return value


class MuJoCoGripperService:
    """Own gripper target mapping, finger torque control, and gripper state queries."""

    def __init__(
        self,
        runtime: MuJoCoRuntimeAdapter,
        finger_indices: Sequence[int],
        *,
        kp_finger: float,
        kd_finger: float,
        finger_bias_mode: str,
    ) -> None:
        self._runtime = runtime
        self._finger_indices = tuple(int(i) for i in finger_indices)
        self._qpos_adr = runtime.qpos_adr
        self._qvel_adr = runtime.qvel_adr
        self._actuator_ids = runtime.actuator_ids
        self._finger_bias_mode = finger_bias_mode
        self._controller = PDController(
            Kp=np.full(len(self._finger_indices), float(kp_finger)),
            Kd=np.full(len(self._finger_indices), float(kd_finger)),
        )

    def set_target_percent(
        self,
        percent: float,
        q_target_desired: Sequence[float],
    ) -> np.ndarray:
        q_target = np.array(q_target_desired, dtype=float).copy()
        p = max(0.0, min(1.0, float(percent)))
        for idx in self._finger_indices:
            low, high = self._runtime.joint_limits[idx]
            if np.isfinite(low) and np.isfinite(high):
                q_target[idx] = low + (1.0 - p) * (high - low)
        return self._clip_joint_targets(q_target)

    def compute_actuator_commands(
        self,
        q_target_desired: Sequence[float],
    ) -> list[tuple[int, float]]:
        data = self._runtime.data
        q_fingers = np.array([data.qpos[self._qpos_adr[i]] for i in self._finger_indices], dtype=float)
        qd_fingers = np.array([data.qvel[self._qvel_adr[i]] for i in self._finger_indices], dtype=float)
        q_fingers_des = np.array([q_target_desired[i] for i in self._finger_indices], dtype=float)

        tau_fingers_pd, _, _ = self._controller.compute(q_fingers, qd_fingers, q_fingers_des)
        tau_fingers = np.array(tau_fingers_pd, dtype=float)
        if self._finger_bias_mode == "full_bias":
            for fi, idx in enumerate(self._finger_indices):
                tau_fingers[fi] += float(data.qfrc_bias[self._qvel_adr[idx]])

        commands: list[tuple[int, float]] = []
        for fi, finger_idx in enumerate(self._finger_indices):
            actuator_id = self._actuator_ids[finger_idx]
            commands.append((actuator_id, self._clip_actuator_force(actuator_id, float(tau_fingers[fi]))))
        return commands

    def get_gripper_state(self, q_target_desired: Sequence[float]) -> dict:
        data = self._runtime.data

        percents: list[float] = []
        target_percents: list[float] = []
        pos_errs: list[float] = []
        vels: list[float] = []

        for idx in self._finger_indices:
            low, high = self._runtime.joint_limits[idx]
            if not (np.isfinite(low) and np.isfinite(high) and high > low):
                continue

            q = float(data.qpos[self._qpos_adr[idx]])
            q_target = float(q_target_desired[idx])
            qd = abs(float(data.qvel[self._qvel_adr[idx]]))

            percent = 1.0 - ((q - low) / (high - low))
            target_percent = 1.0 - ((q_target - low) / (high - low))

            percents.append(float(np.clip(percent, 0.0, 1.0)))
            target_percents.append(float(np.clip(target_percent, 0.0, 1.0)))
            pos_errs.append(abs(q_target - q))
            vels.append(qd)

        max_pos_err = max(pos_errs) if pos_errs else None
        max_vel = max(vels) if vels else None
        percent = float(sum(percents) / len(percents)) if percents else None
        target_percent = float(sum(target_percents) / len(target_percents)) if target_percents else None
        opening = percent is not None and target_percent is not None and target_percent >= percent

        settled = False
        if max_vel is not None:
            if opening:
                settled = bool(max_pos_err is not None and max_pos_err <= 0.05 and max_vel <= 0.2)
            else:
                settled = bool(max_vel <= 0.2)

        return {
            "percent": percent,
            "target_percent": target_percent,
            "max_pos_err": max_pos_err,
            "max_vel": max_vel,
            "settled": settled,
        }

    def wait_for_gripper(
        self,
        *,
        step_fn: Callable[[], None],
        target_provider: Callable[[], Sequence[float]],
        timeout_s: float,
        hold_seconds: float,
        hz: float,
        pos_tol_rad: float,
        vel_tol_rad_s: float,
    ) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        dt = 1.0 / float(hz)
        settled_since: float | None = None

        while time.monotonic() < deadline:
            step_fn()
            state = self.get_gripper_state(target_provider())
            percent = state.get("percent")
            target_percent = state.get("target_percent")
            max_pos_err = state.get("max_pos_err")
            max_vel = state.get("max_vel")
            opening = (
                percent is not None
                and target_percent is not None
                and target_percent >= percent
            )
            if opening:
                settled = bool(
                    max_pos_err is not None
                    and max_vel is not None
                    and max_pos_err <= float(pos_tol_rad)
                    and max_vel <= float(vel_tol_rad_s)
                )
            else:
                settled = bool(max_vel is not None and max_vel <= float(vel_tol_rad_s))

            if settled:
                if settled_since is None:
                    settled_since = time.monotonic()
                if time.monotonic() - settled_since >= float(hold_seconds):
                    return True
            else:
                settled_since = None

            time.sleep(dt)

        return False

    def get_finger_forces(self) -> dict:
        data = self._runtime.data
        f_max = 10.0
        contact_threshold = 0.8 * f_max

        forces = [float(data.actuator_force[self._actuator_ids[idx]]) for idx in self._finger_indices]
        max_abs = max(abs(f) for f in forces) if forces else 0.0

        finger_vels = [
            abs(float(data.qvel[self._qvel_adr[idx]]))
            for idx in self._finger_indices
        ]
        max_vel = max(finger_vels) if finger_vels else 0.0
        contact_detected = max_abs >= contact_threshold and max_vel < 0.1

        return {
            "forces": forces,
            "max_abs_force": round(max_abs, 4),
            "contact_detected": contact_detected,
        }

    def _clip_joint_targets(self, q: np.ndarray) -> np.ndarray:
        q_clipped = q.copy()
        for idx, (low, high) in enumerate(self._runtime.joint_limits):
            if np.isfinite(low) or np.isfinite(high):
                q_clipped[idx] = float(np.clip(q_clipped[idx], low, high))
        return q_clipped

    def _clip_actuator_force(self, actuator_id: int, value: float) -> float:
        model = self._runtime.model
        if model.actuator_forcelimited[actuator_id]:
            low, high = model.actuator_forcerange[actuator_id]
            return float(np.clip(value, low, high))
        return value

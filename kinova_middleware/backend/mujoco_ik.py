from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from kinova_middleware.backend.runtime.mujoco_runtime import MuJoCoRuntimeAdapter

IK_ORIENTATION_WEIGHT = 0.15


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def joint_distance(
    q1: np.ndarray,
    q2: np.ndarray,
    continuous_indices: list[int],
    weights: np.ndarray | None = None,
) -> float:
    """Compute shortest-path joint distance between two configurations."""

    diff = q1 - q2
    for idx in continuous_indices:
        diff[idx] = wrap_to_pi(diff[idx])
    if weights is not None:
        diff = diff * weights
    return float(np.linalg.norm(diff))


def shortest_joint_configuration(
    q_current: np.ndarray,
    q_solution: np.ndarray,
    continuous_indices: list[int],
) -> np.ndarray:
    q_fixed = q_solution.copy()
    for idx in continuous_indices:
        delta = q_solution[idx] - q_current[idx]
        delta_wrapped = wrap_to_pi(delta)
        q_fixed[idx] = q_current[idx] + delta_wrapped
    return q_fixed


def orientation_error_vector(target_vec: np.ndarray, current_vec: np.ndarray) -> np.ndarray:
    """Return the minimal rotation vector needed to align current_vec to target_vec."""

    v0 = current_vec / np.linalg.norm(current_vec)
    v1 = target_vec / np.linalg.norm(target_vec)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    angle = np.arccos(dot)
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.cross(v0, v1)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(v0[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, v0) * v0
        axis = axis / np.linalg.norm(axis)
    else:
        axis = axis / axis_norm
    return axis * angle


class LevenbergMarquardtIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_id: int,
        arm_qpos_adrs: list[int],
        arm_qvel_adrs: list[int],
        arm_joint_limits: list[tuple[float, float]],
        continuous_indices: list[int] | None = None,
        max_iters: int = 50,
        pos_tol: float = 0.001,
        ori_weight: float = IK_ORIENTATION_WEIGHT,
        initial_lambda: float = 0.1,
    ) -> None:
        self.model = model
        self.data = data
        self.ee_site_id = ee_site_id
        self.arm_qpos_adrs = arm_qpos_adrs
        self.arm_qvel_adrs = arm_qvel_adrs
        self.joint_limits = arm_joint_limits
        self.continuous_indices = continuous_indices or []
        self.n_arm = len(arm_qpos_adrs)

        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.ori_weight = ori_weight
        self.initial_lambda = initial_lambda

        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.eye = np.eye(self.n_arm)

        self.stall_threshold = 1e-6
        self.stall_patience = 5
        self.last_iterations = 0

    def solve(
        self,
        target_pos: np.ndarray,
        target_vec: np.ndarray | None = None,
        q_seed: np.ndarray | None = None,
        active_dof: int | None = None,
    ) -> np.ndarray:
        if active_dof is None:
            active_dof = self.n_arm
        target_pos = np.asarray(target_pos, dtype=float)
        if target_vec is not None:
            target_vec = np.asarray(target_vec, dtype=float)
            target_vec = target_vec / np.linalg.norm(target_vec)

        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        q_current = np.array([self.data.qpos[a] for a in self.arm_qpos_adrs], dtype=float)
        seeds = self._build_seeds(q_seed)

        if active_dof < self.n_arm:
            for s in seeds:
                s[active_dof:] = q_current[active_dof:]

        results = []
        for seed in seeds:
            q, iters, perr, oerr = self._solve_single(target_pos, target_vec, seed, active_dof)
            q = shortest_joint_configuration(q_current, q, self.continuous_indices)
            results.append((q.copy(), iters, perr, oerr))

        valid_results = [r for r in results if r[2] < 0.01]
        if valid_results:
            valid_results.sort(key=lambda r: r[3])
            best_ori_err = valid_results[0][3]
            top_results = [r for r in valid_results if r[3] <= best_ori_err + 0.0174]

            j_weights = np.array([5.0, 2.0, 2.0, 1.0])
            top_results.sort(
                key=lambda r: joint_distance(
                    r[0] * j_weights,
                    q_current * j_weights,
                    self.continuous_indices,
                )
            )
            best_q, best_iters, best_pos_err, best_ori_err = top_results[0]
        else:
            results.sort(key=lambda r: r[2])
            best_q, best_iters, best_pos_err, best_ori_err = results[0]

        self.last_iterations = best_iters

        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)

        return best_q

    def _solve_single(
        self,
        target_pos: np.ndarray,
        target_vec: np.ndarray | None,
        q_start: np.ndarray,
        active_dof: int,
    ) -> tuple[np.ndarray, int, float, float]:
        q_arm = q_start.copy()
        for i_arm, adr in enumerate(self.arm_qpos_adrs):
            self.data.qpos[adr] = q_arm[i_arm]
        mujoco.mj_forward(self.model, self.data)

        lam = self.initial_lambda
        v = 2.0
        best_err = float("inf")
        stalls = 0

        for it in range(self.max_iters):
            ee_pos = np.array(self.data.site_xpos[self.ee_site_id], dtype=float)
            ee_xmat = np.array(self.data.site_xmat[self.ee_site_id], dtype=float).reshape(3, 3)
            current_vec = ee_xmat[:, 0]

            pos_err_vec = target_pos - ee_pos
            pos_err_norm = float(np.linalg.norm(pos_err_vec))

            if target_vec is not None:
                ori_err_vec = orientation_error_vector(target_vec, current_vec)
                ori_err_norm = float(np.linalg.norm(ori_err_vec))
                residual = np.concatenate([pos_err_vec, self.ori_weight * ori_err_vec])
                cost = 0.5 * np.sum(residual**2)
            else:
                ori_err_norm = 0.0
                residual = pos_err_vec
                cost = 0.5 * np.sum(residual**2)

            if target_vec is None:
                if pos_err_norm < self.pos_tol:
                    return q_arm, it + 1, pos_err_norm, 0.0
            else:
                if pos_err_norm < self.pos_tol and ori_err_norm < 0.05:
                    return q_arm, it + 1, pos_err_norm, ori_err_norm

            if cost >= best_err:
                lam *= v
                v *= 2.0
            else:
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * (best_err - cost) / cost) ** 3)
                v = 2.0
                best_err = cost

            if cost > 0 and abs(best_err - cost) < self.stall_threshold:
                stalls += 1
                if stalls >= self.stall_patience:
                    break
            else:
                stalls = 0

            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.ee_site_id)
            cols = self.arm_qvel_adrs[:active_dof]
            J = self.jacp[:, cols]
            if target_vec is not None:
                J = np.vstack([J, self.ori_weight * self.jacr[:, cols]])

            H = J.T @ J
            g = J.T @ residual
            H_lm = H + lam * np.eye(active_dof)

            import numpy.linalg as nla

            try:
                dq = nla.solve(H_lm, g)
            except nla.LinAlgError:
                dq = nla.lstsq(H_lm, g, rcond=None)[0]

            q_arm[:active_dof] = q_arm[:active_dof] + dq

            for j in range(self.n_arm):
                low, high = self.joint_limits[j]
                if np.isfinite(low) or np.isfinite(high):
                    q_arm[j] = float(np.clip(q_arm[j], low, high))

            for i_arm, adr in enumerate(self.arm_qpos_adrs):
                self.data.qpos[adr] = q_arm[i_arm]
            mujoco.mj_forward(self.model, self.data)

        ee_pos = np.array(self.data.site_xpos[self.ee_site_id], dtype=float)
        ee_xmat = np.array(self.data.site_xmat[self.ee_site_id], dtype=float).reshape(3, 3)
        pos_err_norm = float(np.linalg.norm(target_pos - ee_pos))
        ori_err_norm = 0.0
        if target_vec is not None:
            ori_err_vec = orientation_error_vector(target_vec, ee_xmat[:, 0])
            ori_err_norm = float(np.linalg.norm(ori_err_vec))

        return q_arm, self.max_iters, pos_err_norm, ori_err_norm

    def _build_seeds(self, q_seed: np.ndarray | None) -> list[np.ndarray]:
        q_current = np.array([self.data.qpos[a] for a in self.arm_qpos_adrs], dtype=float)
        y0 = float(q_current[0])
        yaw_bases = [y0, y0 + 1.57, y0 - 1.57, y0 + 3.14]
        if q_seed is not None:
            y_seed = float(q_seed[0])
            yaw_bases = [y_seed, y_seed + 1.57, y_seed - 1.57, y_seed + 3.14]
        yaw_bases = [(y + 3.14159) % (2 * 3.14159) - 3.14159 for y in yaw_bases]

        pitch_vals = [1.5, -1.5, 3.14, -3.14]
        seeds = []
        for yaw in yaw_bases:
            for p1 in pitch_vals:
                for p2 in pitch_vals:
                    seed = q_current.copy()
                    seed[0], seed[1], seed[2] = yaw, p1, p2
                    seeds.append(seed)

        import random as random_module

        for _ in range(50):
            seed = q_current.copy()
            for j in range(self.n_arm):
                low, high = self.joint_limits[j]
                if np.isfinite(low) and np.isfinite(high):
                    seed[j] = random_module.uniform(low, high)
                else:
                    seed[j] = random_module.uniform(-3.14, 3.14)
            seeds.append(seed)

        if q_seed is not None:
            seeds.insert(0, np.array(q_seed[: self.n_arm], dtype=float))

        return seeds


class MuJoCoIKService:
    """Backend-facing IK service that depends only on the MuJoCo runtime adapter."""

    def __init__(
        self,
        runtime: MuJoCoRuntimeAdapter,
        arm_indices: Sequence[int],
        *,
        max_iters: int = 50,
        pos_tol: float = 0.001,
        ori_weight: float = IK_ORIENTATION_WEIGHT,
        initial_lambda: float = 0.1,
    ) -> None:
        self._runtime = runtime
        self._arm_indices = tuple(int(i) for i in arm_indices)
        self._max_iters = int(max_iters)
        self._pos_tol = float(pos_tol)
        self._ori_weight = float(ori_weight)
        self._initial_lambda = float(initial_lambda)
        self._solver: LevenbergMarquardtIK | None = None
        self.last_iterations: int | None = None

    def solve(
        self,
        target_pos: Sequence[float],
        target_vec: Sequence[float] | None = None,
        q_seed: Sequence[float] | None = None,
        active_dof: int | None = None,
    ) -> np.ndarray:
        solver = self._get_solver()
        seed_array = None if q_seed is None else np.asarray(q_seed, dtype=float)
        target_vec_array = None if target_vec is None else np.asarray(target_vec, dtype=float)
        best_q = solver.solve(
            np.asarray(target_pos, dtype=float),
            target_vec_array,
            seed_array,
            active_dof=active_dof,
        )
        self.last_iterations = solver.last_iterations
        return best_q

    def reset(self) -> None:
        self._solver = None
        self.last_iterations = None

    def _get_solver(self) -> LevenbergMarquardtIK:
        if self._solver is None:
            ee_site_id = self._runtime.ee_site_id
            if ee_site_id is None:
                raise RuntimeError("MuJoCo IK service requires an end-effector site.")

            arm_continuous_indices = [
                i_arm
                for i_arm, idx_full in enumerate(self._arm_indices)
                if idx_full in self._runtime.continuous_indices
            ]

            self._solver = LevenbergMarquardtIK(
                model=self._runtime.model,
                data=self._runtime.data,
                ee_site_id=ee_site_id,
                arm_qpos_adrs=[self._runtime.qpos_adr[i] for i in self._arm_indices],
                arm_qvel_adrs=[self._runtime.qvel_adr[i] for i in self._arm_indices],
                arm_joint_limits=[self._runtime.joint_limits[i] for i in self._arm_indices],
                continuous_indices=arm_continuous_indices,
                max_iters=self._max_iters,
                pos_tol=self._pos_tol,
                ori_weight=self._ori_weight,
                initial_lambda=self._initial_lambda,
            )
        return self._solver

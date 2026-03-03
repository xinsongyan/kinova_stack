"""Jerk-limited trajectory generator with braking feasibility."""

from __future__ import annotations

import numpy as np


class TrajectoryGenerator:
    """Produces (q_des, qd_des, qdd_des) with jerk/accel/vel clamping.

    Uses a trapezoidal velocity profile (accelerate → cruise → brake)
    rather than a 2nd-order filter.  This gives predictable, near-time-
    optimal motion within jerk/accel/vel limits.

    Key design choices:

    1. **Braking feasibility** — Before integrating, check whether the
       joint can stop within the remaining distance at its current velocity.
       If not, force braking.

    2. **Jerk clamping** — The rate-of-change of acceleration is bounded
       by j_max, ensuring smooth torque transitions.

    3. **Near-target snap** — When close with low vel/accel, snap to target.
    """

    def __init__(
        self,
        n_joints: int,
        v_max: np.ndarray | list[float],
        a_max: np.ndarray | list[float],
        j_max: np.ndarray | list[float],
        dt: float,
        omega: np.ndarray | list[float] | None = None,   # unused, kept for API compat
    ) -> None:
        self.n = n_joints
        self.dt = dt

        # Nominal limits (immutable reference for governor recovery)
        self._v_max_nom = np.array(v_max, dtype=float)
        self._a_max_nom = np.array(a_max, dtype=float)

        # Active limits (governor can scale these down)
        self.v_max = np.array(v_max, dtype=float)
        self.a_max = np.array(a_max, dtype=float)
        self.j_max = np.array(j_max, dtype=float)

        # Internal state
        self.q = np.zeros(n_joints, dtype=float)
        self.qd = np.zeros(n_joints, dtype=float)
        self.qdd = np.zeros(n_joints, dtype=float)
        self.q_goal = np.zeros(n_joints, dtype=float)

        # Diagnostics
        self.braking = np.zeros(n_joints, dtype=bool)

    def reset(self, q_current: np.ndarray) -> None:
        self.q[:] = q_current
        self.qd[:] = 0.0
        self.qdd[:] = 0.0
        self.q_goal[:] = q_current
        self.braking[:] = False

    def set_goal(self, q_goal: np.ndarray) -> None:
        self.q_goal[:] = q_goal

    def set_limits(self, joint_idx: int, *, v_max: float, a_max: float) -> None:
        self.v_max[joint_idx] = v_max
        self.a_max[joint_idx] = a_max

    def update(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance one control tick.  Returns (q_des, qd_des, qdd_des)."""
        dt = self.dt
        for i in range(self.n):
            err = self.q_goal[i] - self.q[i]
            sign_err = np.sign(err) if abs(err) > 1e-6 else 0.0
            abs_err = abs(err)
            abs_qd = abs(self.qd[i])

            # --- Stopping distance at current speed ---
            d_stop = (self.qd[i] ** 2) / (2.0 * max(self.a_max[i], 1e-6))

            # --- Decide: accelerate, cruise, or brake ---
            if abs_err <= d_stop * 1.05 and abs_qd > 1e-3:
                # Must brake to stop at goal
                self.braking[i] = True
                a_desired = -np.sign(self.qd[i]) * self.a_max[i]
            elif abs_qd >= self.v_max[i] - 1e-4:
                # At cruise velocity, just hold
                self.braking[i] = False
                a_desired = 0.0
            else:
                # Accelerate toward goal
                self.braking[i] = False
                a_desired = sign_err * self.a_max[i]

            # --- Jerk limit ---
            da = a_desired - self.qdd[i]
            j_lim = self.j_max[i] * dt
            da = float(np.clip(da, -j_lim, j_lim))
            self.qdd[i] += da

            # --- Acceleration clamp ---
            self.qdd[i] = float(np.clip(self.qdd[i], -self.a_max[i], self.a_max[i]))

            # --- Integrate velocity + clamp ---
            self.qd[i] += self.qdd[i] * dt
            self.qd[i] = float(np.clip(self.qd[i], -self.v_max[i], self.v_max[i]))

            # --- Integrate position ---
            self.q[i] += self.qd[i] * dt

            # --- Overshoot prevention: if we passed the goal, clamp ---
            new_err = self.q_goal[i] - self.q[i]
            if err * new_err < 0:  # sign changed → overshot
                self.q[i] = self.q_goal[i]
                self.qd[i] = 0.0
                self.qdd[i] = 0.0

            # --- Near-target snap ---
            if abs_err < 1e-4 and abs_qd < 1e-3 and abs(self.qdd[i]) < 0.1:
                self.q[i] = self.q_goal[i]
                self.qd[i] = 0.0
                self.qdd[i] = 0.0

        return self.q.copy(), self.qd.copy(), self.qdd.copy()

"""Controllers for MuJoCo torque-controlled robots."""

from __future__ import annotations

import mujoco
import numpy as np


class PDController:
    """Simple PD controller (kept for finger joints)."""

    def __init__(self, Kp, Kd):
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)

    def compute(self, q, qd, q_desired, qd_desired=None):
        if qd_desired is None:
            qd_desired = np.zeros_like(q)
        P = self.Kp * (q_desired - q)
        D = self.Kd * (qd_desired - qd)
        return P + D, P, D


class ComputedTorqueController:
    """Inverse-dynamics tracking controller using mj_fullM + qfrc_bias.

    Computes:
        qdd_cmd = qdd_des + Kp·(q_des − q) + Kd·(qd_des − qd)
        tau = M(q) @ qdd_cmd_full + qfrc_bias

    Uses mujoco.mj_fullM to get the dense mass matrix and data.qfrc_bias
    for gravity + Coriolis + centrifugal forces.  Does NOT mutate data.qacc
    or call mj_rne, so the physics state is never corrupted.

    MuJoCo API notes:
    - mj_fullM expects a flat float64 array of length nv*nv. It fills the
      LOWER triangle of the symmetric mass matrix from the compressed
      data.qM. We symmetrise it ourselves.
    - qfrc_bias is kept up-to-date by mj_step (via mj_forward), so it is
      always consistent with the current (qpos, qvel).
    """

    def __init__(
        self,
        Kp: np.ndarray,
        Kd: np.ndarray,
        arm_dof_adrs: list[int],
        nv: int,
    ) -> None:
        self.Kp = np.array(Kp, dtype=float)
        self.Kd = np.array(Kd, dtype=float)
        self.arm_dof_adrs = list(arm_dof_adrs)
        self.nv = nv
        # Pre-allocate (nv, nv) buffer for mj_fullM.
        # MuJoCo Python bindings require shape (nv, nv), not flat.
        self._M = np.zeros((nv, nv), dtype=np.float64)

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_des: np.ndarray,
        qd_des: np.ndarray,
        qdd_des: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute torque for arm joints.

        Returns:
            tau_arm:  (n_arm,) complete torque command (M·qdd_cmd + bias)
            qdd_pd:  (n_arm,) acceleration-space PD correction only
            qdd_cmd: (n_arm,) total commanded acceleration (feedforward + PD)
        """
        nv = self.nv

        # --- PD correction in acceleration space ---
        e = q_des - q
        ed = qd_des - qd
        qdd_pd = self.Kp * e + self.Kd * ed
        qdd_cmd = qdd_des + qdd_pd

        # --- Build full-nv qacc vector (arm entries only, rest zero) ---
        qacc_full = np.zeros(nv, dtype=np.float64)
        for i, adr in enumerate(self.arm_dof_adrs):
            qacc_full[adr] = qdd_cmd[i]

        # --- Mass matrix via mj_fullM (no mutation of data.qacc) ---
        # mj_fullM fills a symmetric (nv, nv) matrix from data.qM.
        M = self._M
        mujoco.mj_fullM(model, M, data.qM)

        # --- tau = M @ qacc_full + qfrc_bias ---
        # qfrc_bias contains gravity + Coriolis/centrifugal but NOT
        # joint damping (dof_damping * qvel) or friction (dof_frictionloss).
        # MuJoCo applies damping/friction internally during mj_step, so
        # we must add feedforward compensation here.
        bias = np.array(data.qfrc_bias, dtype=np.float64)
        tau_full = M @ qacc_full + bias

        # --- Damping + friction feedforward for arm DOFs ---
        # Instead of canceling current damping (which leaves PD to do all the work
        # of accelerating the arm against friction), we feedforward the torque
        # needed to maintain the *desired* trajectory velocity.
        for i, adr in enumerate(self.arm_dof_adrs):
            damp = float(model.dof_damping[adr])
            fric = float(model.dof_frictionloss[adr])
            # Feedforward torque to overcome damping at desired velocity
            tau_full[adr] += damp * qd_des[i]
            # Feedforward torque to overcome friction if we want to move
            if abs(qd_des[i]) > 1e-4:
                tau_full[adr] += fric * np.sign(qd_des[i])

        # --- Extract arm DOFs ---
        tau_arm = np.array([tau_full[adr] for adr in self.arm_dof_adrs], dtype=float)

        return tau_arm, qdd_pd, qdd_cmd

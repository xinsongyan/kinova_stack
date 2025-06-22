import numpy as np
import mujoco


class PDController:
    """Simple joint-space PD torque controller."""

    def __init__(self, Kp: np.ndarray, Kd: np.ndarray):
        self.Kp = np.asarray(Kp)
        self.Kd = np.asarray(Kd)

    def __call__(self, q, qd, q_des, qd_des=None):
        if qd_des is None:
            qd_des = np.zeros_like(q)
        return self.Kp * (q_des - q) + self.Kd * (qd_des - qd)


class JointSpaceImpedance:
    """Critically damped impedance controller in joint space."""

    def __init__(self, stiffness, damping=None):
        self.k = np.asarray(stiffness)
        self.d = np.asarray(damping) if damping is not None else 2 * np.sqrt(self.k)

    def __call__(self, q, qd, q_des):
        return self.k * (q_des - q) - self.d * qd


class ArmPositionController:
    """
    4-DoF PD + gravity compensation.

    Parameters
    ----------
    kp, kd       : gains (scalar or length-4 array)
    torque_limit : None or scalar  – per-joint clamp
    alpha        : 1.0 -> step reference, <1 -> internal first-order filter
    """
    IDX = slice(0, 4) # arm joints

    def __init__(self, kp=120, kd=None, torque_limit=None, alpha=1.0):
        self.kp = np.full(4, kp) if np.isscalar(kp) else np.asarray(kp, float)
        self.kd = (np.sqrt(self.kp) * 0.8 if kd is None
                   else np.asarray(kd, float))
        self.tau_max = torque_limit
        self.alpha   = float(alpha)
        self._q_des_filt = None # internal filtered set-point

    def __call__(self, data, q_des, qd_des=None):
        q   = data.qpos[self.IDX]
        qd  = data.qvel[self.IDX]

        q_des = np.asarray(q_des, float)
        if qd_des is None:
            qd_des = np.zeros_like(q_des)

        # 1st-order filter on the desired position
        if self._q_des_filt is None:
            self._q_des_filt = q_des.copy()
        else:
            self._q_des_filt = (self.alpha * q_des + (1 - self.alpha) * self._q_des_filt)

        # PD control torque
        tau_pd = (self.kp * (self._q_des_filt - q) + self.kd * (qd_des - qd))

        # Gravity / Coriolis compensation from MuJoCo
        # tau_bias = data.qfrc_bias[self.IDX]
        # tau = tau_pd + tau_bias
        tau = tau_pd

        if self.tau_max is not None:
            tau = np.clip(tau, -self.tau_max, self.tau_max)

        return tau # (4,)

# --------- Hand controller ---------
class HandPositionController: #TODO: test
    """6-DoF finger PD controller (3 proximal + 3 distal)."""
    def __init__(self, kp=10, kd=None, torque_limit=0.5):
        self.kp = np.full(6, kp) if np.isscalar(kp) else np.asarray(kp, float)
        self.kd = np.sqrt(self.kp) * 0.8 if kd is None else np.asarray(kd, float)
        self.tau_max = torque_limit
        # self.tau_max = None

    def __call__(self, q_hand, qd_hand, q_des, qd_des=None):
        if qd_des is None:
            qd_des = np.zeros_like(q_des)
        tau = self.kp * (q_des - q_hand) + self.kd * (qd_des - qd_hand)
        if self.tau_max is not None:
            tau = np.clip(tau, -self.tau_max, self.tau_max)

        assert tau.shape == (6,), "Expected 6-element torque vector for hand controller."
        return tau


# --------- Task-Space Controller ---------
class OperationalSpaceController: #TODO: test
    """
    Torque-level operational-space controller (position+orientation) of the arm
    Uses damped-LS to invert the 6×N Jacobian.
    """
    def __init__(self, model, kp_xyz=400, kp_ori=150, damping=1e-3):
        self.m = model
        self.kp_xyz = kp_xyz
        self.kp_ori = kp_ori
        self.damping = damping

    def _jacobian(self, data, body_id):
        """Return 6×N Jacobian of *body* in world frame."""
        if body_id < 0:
            raise ValueError("Invalid body_id passed to _jacobian().")
        nv = self.m.nv
        Jp = np.zeros((3, nv), order='C')
        Jr = np.zeros((3, nv), order='C')
        mujoco.mj_jacBody(self.m, data, Jp, Jr, body_id)
        return np.vstack((Jp, Jr))
    
    def _jac_pos(self, data, body_id):
        """Return 3×N Jacobian of *body* in world frame."""
        nv  = self.m.nv
        Jp  = np.zeros((3, nv), order='C')
        Jr  = np.zeros((3, nv), order='C')      # dummy (ignored)
        mujoco.mj_jacBody(self.m, data, Jp, Jr, body_id)
        return Jp

    def __call__(self, data, body_id, pos_d, quat_d, pos_only=False):
        x    = data.xpos[body_id].copy()
        quat = data.xquat[body_id].copy()

        # --- position error
        e_pos = pos_d - x # shape (3,)

        # --- orientation error
        e_ori = np.zeros(3) # buffer for 3-vector
        
        mujoco.mju_subQuat(e_ori, quat_d, quat) # this is the quaternion error

        if pos_only:
            # If only position control is required, return the torque for position error only            
            f = self.kp * e_pos # shape (3,)

            # Jacobian (position only)
            J = self._jac_pos(data, body_id)
            JT = J.T
            JJ = J @ JT
            tau = JT @ np.linalg.solve(JJ + self.lmbda * np.eye(3), f)

        else:
            # Task-space PD wrench
            fx = np.hstack((self.kp_xyz * e_pos, self.kp_ori * e_ori)) # shape (6,)

            J  = self._jacobian(data, body_id) # 6×N
            JT = J.T
            JJ = J @ JT
            damping = self.damping * np.eye(6)
            tau = JT @ np.linalg.solve(JJ + damping, fx) # shape (N,); use damped LS inversion 

        # Check: the singular value of J
        sv = np.linalg.svd(J, compute_uv=False)
        rank = np.linalg.matrix_rank(J)
        # print(f"Operational Space Controller: Singular values of J: {sv}, rank of J: {rank}")
        assert tau.shape[0] == self.m.nu, "Expected torque vector to match model's number of actuators."
        return tau
    

# --------- Helpers ---------
def _as_np(x, dtype=float):
    return np.asarray(x, dtype=dtype, order='C')
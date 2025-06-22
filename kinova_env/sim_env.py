from __future__ import annotations
import sys
import numpy as np
import mujoco
import mujoco.viewer

# For WSL
import warnings
warnings.filterwarnings("ignore", message=".*Wayland: The platform does not provide the window position.*")


def get_test_model_path():
    import importlib.resources
    with importlib.resources.path("mujoco", "testdata") as model_dir:
        model_path = model_dir / "model.xml"
        return str(model_path)
    

class SimEnv:
    def __init__(self, model_xml: str, viewer: bool = True):
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.nu = self.model.nu

        self._viewer = (
            mujoco.viewer.launch_passive(self.model, self.data)
            if viewer else None
        )

    # ---------- MuJoCo helpers
    def _key_id(self, name: str) -> int:
        """Return the numeric id of the keyframe called *name*."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id < 0:
            raise ValueError(f"Keyframe '{name}' not found in model.")
        return key_id

    # ---------- API
    def set_keyframe(self, name: str):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key_id(name))

    def get_keyframe_qpos(self, name: str):
        kid = self._key_id(name)
        return np.array(self.model.key_qpos[kid])
    
    # ---------- EE helpers
    def ee_body_id(self):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_4")
        # print(f"EE body ID: {bid}")
        if bid < 0:
            raise RuntimeError("Body 'link_4' not found in the MJCF.")
        return bid

    def ee_pose(self):
        bid = self.ee_body_id()
        return (self.data.xpos[bid].copy(), self.data.xquat[bid].copy())
    

    def print_ee_pose(self, degrees: bool = False):
        pos, quat = self.ee_pose()
        if degrees:
            euler = np.degrees(mujoco.mju_quat2Mat(quat).rpy()) # roll-pitch-yaw
            ori = f"RPY(deg)={euler.round(1)}"
        else:
            ori = f"quat={quat.round(3)}"
        print(f"EE pos [m]: {pos.round(3)}   {ori}")

    # ---------- simulation loop ----------
    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            if self._viewer.is_running():
                self._viewer.sync()
            else:
                sys.exit("Viewer closed. Exiting.")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    # ---------- state  ----------
    def qpos(self):
        return np.copy(self.data.qpos)

    def qvel(self):
        return np.copy(self.data.qvel)

    def state(self):
        return self.qpos(), self.qvel()

    def time(self):
        return self.data.time

    # ---------- command ----------
    def set_torque(self, tau):
        self.data.ctrl[:] = tau


    # ---------- incremental Cartesian move ----------
    def move_ee_delta(self, delta_xyz, duration=5.0, kp_xyz=400, kp_ori=150):
        """
        Move the EE by (dx,dy,dz) in world frame while keeping orientation.
        *duration* is in seconds.
        """
        from kinova_control.controllers import OperationalSpaceController

        delta = np.asarray(delta_xyz, float)
        steps = int(duration / self.dt)
        pos_d, quat_d = self.ee_pose()
        pos_d = pos_d + delta # new target
        osc = OperationalSpaceController(self.model, kp_xyz=kp_xyz, kp_ori=kp_ori)
        bid = self.ee_body_id()

        for _ in range(steps):
            tau  = osc(self.data, bid, pos_d, quat_d)
            self.data.ctrl[:4] = tau[:4]    # arm motors only
            self.data.ctrl[4:] = 0.0
            mujoco.mj_step(self.model, self.data)
            if self._viewer: self._viewer.sync()

    # ---------- move one finger joint ----------
    def move_finger_joint(self, finger: int, joint: int, delta_deg: float, duration=0.5, kp=8):
        """
        finger: {1,2,3}, joint: {1 (proximal), 2 (distal)}.
        Positive *delta_deg* closes the finger further.
        """
        from kinova_control.controllers import HandPositionController

        if finger not in (1, 2, 3) or joint not in (1, 2):
            raise ValueError("finger must be 1-3, joint 1(prox) or 2(distal)")

        # joint index table inside qpos[4:10]
        idx_map = {(1, 1): 0, (2, 1): 1, (3, 1): 2,
                   (1, 2): 3, (2, 2): 4, (3, 2): 5}
        local_idx = idx_map[(finger, joint)] # 0-5 inside hand slice
        global_idx = 4 + local_idx # 0-9 in full qpos

        steps = int(duration / self.dt)
        q_des = self.qpos()[4:10].copy()
        q_des[local_idx] += np.deg2rad(delta_deg)

        hand_ctr = HandPositionController(kp=kp)

        for _ in range(steps):
            q, qd = self.state()
            tau_hand = hand_ctr(q[4:10], qd[4:10], q_des)
            self.data.ctrl[:4]  = 0.0 # keep arm passive
            self.data.ctrl[4:] = tau_hand
            mujoco.mj_step(self.model, self.data)
            if self._viewer: self._viewer.sync()


    def ee_at_position(self, target_xyz, tol=1e-3):
        """
        Check if the end-effector is at the target position within a tolerance.
        Example: sim.ee_at_position([0.4, 0.0, 0.25], tol=2e-3)
        """
        target = np.asarray(target_xyz, float)
        pos, _ = self.ee_pose()
        return np.linalg.norm(pos - target) <= tol, np.linalg.norm(pos - target)


    def joints_close(self, q_desired, tol=1e-3, indices=None):
        """Check if the current joint positions are close to the desired ones."""
        q_desired = np.asarray(q_desired, float)
        tol = np.asarray(tol, float)
        q_curr = self.qpos()

        if indices is not None:
            q_curr = q_curr[indices]
            q_desired = q_desired[indices]

        return np.all(np.abs(q_curr - q_desired) <= tol), np.abs(q_curr - q_desired)


if __name__ == "__main__":
    # Example usage
    model_path = get_test_model_path()  
    env = SimEnv(model_path)

    while True:
        env.step()
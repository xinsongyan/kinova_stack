"""Demo: full 360deg rotation of Joint 1 (headless tuning).

Run from project root:  mjpython kinova_sim/main.py
"""

import os
import math
import sys
import numpy as np
import mujoco
import mujoco.viewer

from controller import ComputedTorqueController
from trajectory import TrajectoryGenerator

# Resolve paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF_PATH = os.path.realpath(os.path.join(
    _HERE, "..", "kinova_description", "mjcf", "m1n4s300_standalone.mjcf"
))
_MESH_DIR = os.path.realpath(os.path.join(
    _HERE, "..", "kinova_description", "meshes"
))

ARM_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4")

# Home position (from scene XMLs — not in standalone MJCF)
# qpos order: joint_1, joint_2, joint_3, joint_4, then 6 finger joints
HOME_QPOS = [1.48353, 3.22886, 4.90438, 0.0,  0, 0, 0, 0, 0, 0]

# --- Tuning to keep J1 tau under ±5 Nm ---
# We use v_max = 0.8 rad/s. Damping feedforward takes 2.0*0.8 = 1.6 Nm.
# Friction feedforward takes 0.5 Nm. 
# Total feedforward = 2.1 Nm, leaving 2.9 Nm available for acceleration!
V_MAX = np.array([1.0, 2.0, 2.0, 2.0])
A_MAX = np.array([2.0, 15.0, 15.0, 15.0])
J_MAX = np.array([20.0, 150.0, 150.0, 150.0])

# Strong PD gains for J1 to quickly overcome static friction near target,
# with moderate KD to avoid noise amplification limit-cycling.
# High J2/J3/J4 gains to firmly hold the folded arm against gravity.
KP = np.array([500.0, 800.0, 800.0, 400.0])
KD = np.array([10.0,  100.0, 100.0,  50.0])

CONTROL_DT = 0.001


def _load_model(mjcf_path, mesh_dir):
    with open(mjcf_path, "r") as f:
        xml = f.read()
    xml = xml.replace("<compiler ", f'<compiler meshdir="{mesh_dir}" ', 1)
    
    # Remove the ground plane so the stowed arm doesn't catch on the floor
    import re
    xml = re.sub(r'<geom name="ground".*?/>', '', xml, flags=re.DOTALL)
    
    # Disable collision on the base mesh since it's in the world body and collides with the shoulder
    xml = xml.replace('<geom type="mesh" mesh="base" material="black"/>', 
                      '<geom type="mesh" mesh="base" material="black" contype="0" conaffinity="0"/>')
    
    return mujoco.MjModel.from_xml_string(xml)


def _step_n(model, data, viewer, n, realtime=True):
    for _ in range(n):
        mujoco.mj_step(model, data)
    if viewer is not None:
        if viewer.is_running():
            viewer.sync()
        else:
            import sys
            sys.exit("Viewer closed.")
    if realtime:
        import time
        time.sleep(n * model.opt.timestep)


def main():
    model = _load_model(_MJCF_PATH, _MESH_DIR)
    data = mujoco.MjData(model)
    
    viewer = mujoco.viewer.launch_passive(model, data)

    # Home position
    data.qpos[:len(HOME_QPOS)] = HOME_QPOS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    arm_jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
    arm_dof_adrs = [int(model.jnt_dofadr[jid]) for jid in arm_jnt_ids]
    arm_qpos_adrs = [int(model.jnt_qposadr[jid]) for jid in arm_jnt_ids]

    physics_dt = model.opt.timestep
    n_substeps = max(1, math.ceil(CONTROL_DT / physics_dt))
    control_dt = n_substeps * physics_dt

    q0 = np.array([data.qpos[a] for a in arm_qpos_adrs], dtype=float)

    traj = TrajectoryGenerator(n_joints=4, v_max=V_MAX, a_max=A_MAX, j_max=J_MAX, dt=control_dt)
    traj.reset(q0)
    ct = ComputedTorqueController(Kp=KP, Kd=KD, arm_dof_adrs=arm_dof_adrs, nv=model.nv)

    # Goal: +360
    q_goal = q0.copy()
    q_goal[0] += 2 * math.pi
    
    j1_limit = 5.0
    actuator_ids = {jn: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{jn}")
                    for jn in ARM_JOINTS}

    print(f"Demo: J1 360 deg rotation ({math.degrees(q0[0]):.0f} -> {math.degrees(q_goal[0]):.0f})")
    print(f"Tracking within strictly ±5.0 Nm limitation")
    print(f"Waiting 1s before starting...\n")

    # Hold at home 1s
    wait_end = data.time + 1.0
    while data.time < wait_end:
        q_arm = np.array([data.qpos[a] for a in arm_qpos_adrs], dtype=float)
        qd_arm = np.array([data.qvel[a] for a in arm_dof_adrs], dtype=float)
        q_des, qd_des, qdd_des = traj.update()
        tau_arm, _, _ = ct.compute(model, data, q_des, qd_des, qdd_des, q_arm, qd_arm)
        tau_arm[0] = j1_limit * np.tanh(tau_arm[0] / j1_limit)
        ctrl = np.zeros(model.nu, dtype=float)
        for i, jname in enumerate(ARM_JOINTS):
            ctrl[actuator_ids[jname]] = tau_arm[i]
        data.ctrl[:] = ctrl
        _step_n(model, data, viewer, n_substeps)

    traj.set_goal(q_goal)

    tick = 0
    max_tau_j1 = 0.0
    while True:
        q_arm = np.array([data.qpos[a] for a in arm_qpos_adrs], dtype=float)
        qd_arm = np.array([data.qvel[a] for a in arm_dof_adrs], dtype=float)

        q_des, qd_des, qdd_des = traj.update()
        tau_arm, _, _ = ct.compute(model, data, q_des, qd_des, qdd_des, q_arm, qd_arm)

        tau_arm[0] = j1_limit * np.tanh(tau_arm[0] / j1_limit)
        if abs(tau_arm[0]) > max_tau_j1:
            max_tau_j1 = abs(tau_arm[0])

        ctrl = np.zeros(model.nu, dtype=float)
        for i, jname in enumerate(ARM_JOINTS):
            ctrl[actuator_ids[jname]] = tau_arm[i]
        data.ctrl[:] = ctrl

        _step_n(model, data, viewer, n_substeps)

        tick += 1
        if tick % 1000 == 0:
            err = q_goal[0] - q_arm[0]
            print(f"  t={data.time:5.1f}s  J1={math.degrees(q_arm[0]):6.1f}  "
                  f"err={math.degrees(err):+6.1f} deg  vel={qd_arm[0]:+5.2f} rad/s  "
                  f"tau={tau_arm[0]:+5.2f} Nm  brk={int(traj.braking[0])}")

        if abs(q_goal[0] - q_arm[0]) < math.radians(1.0) and abs(qd_arm[0]) < 0.05:
            print(f"\n  Done! t={data.time:.2f}s  max_tau={max_tau_j1:.2f} Nm")
            
            # keep viewer open
            print("  Close viewer to exit.")
            while viewer.is_running():
                _step_n(model, data, viewer, n_substeps)
            return

    print(f"\n  Timeout at t={data.time:.2f}s  J1={math.degrees(q_arm[0]):.1f}")


if __name__ == "__main__":
    main()
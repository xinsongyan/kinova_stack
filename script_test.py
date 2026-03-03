"""Demo: full 360deg rotation of Joint 1 (headless tuning)."""

import os
import math
import numpy as np
import mujoco

from kinova_sim.controller import ComputedTorqueController
from kinova_sim.trajectory import TrajectoryGenerator

_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF_PATH = os.path.realpath(os.path.join(
    _HERE, "kinova_description", "mjcf", "m1n4s300_standalone.mjcf"
))
_MESH_DIR = os.path.realpath(os.path.join(
    _HERE, "kinova_description", "meshes"
))

ARM_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4")
HOME_QPOS = [1.48353, 3.22886, 4.90438, 0.0, 0, 0, 0, 0, 0, 0]

# --- Tuning: keep J1 tau strictly under 5 Nm ---
# damping=2.0 → at v=1.0 rad/s needs 2.0 Nm for damping + 0.5 friction = 2.5 Nm.
# This leaves 2.5 Nm available for acceleration.
V_MAX = np.array([1.0, 2.0, 2.0, 2.0])
A_MAX = np.array([2.0, 15.0, 15.0, 15.0])
J_MAX = np.array([20.0, 150.0, 150.0, 150.0])

# We use moderate KP (100) and KD (10) for J1 so the PD torque doesn't wildly chat.
# J2, J3, J4 need high gains to hold their posture against gravity!
KP = np.array([100.0, 800.0, 800.0, 400.0])
KD = np.array([10.0, 100.0, 100.0,  50.0])

J1_LIMIT = 5.0   # hard cap at ±5 Nm
CONTROL_DT = 0.001


def _load_model(mjcf_path, mesh_dir):
    with open(mjcf_path, "r") as f:
        xml = f.read()
    xml = xml.replace("<compiler ", f'<compiler meshdir="{mesh_dir}" ', 1)
    
    # Remove the ground plane so the arm doesn't drag on the floor when folded
    import re
    xml = re.sub(r'<geom name="ground".*?/>', '', xml)
    
    return mujoco.MjModel.from_xml_string(xml)


def main():
    model = _load_model(_MJCF_PATH, _MESH_DIR)
    data = mujoco.MjData(model)

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

    actuator_ids = {jn: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{jn}")
                    for jn in ARM_JOINTS}

    # Goal: +360
    q_goal = q0.copy()
    q_goal[0] += 2 * math.pi
    traj.set_goal(q_goal)

    print(f"J1: {math.degrees(q0[0]):.1f} -> {math.degrees(q_goal[0]):.1f}")
    print(f"J1_LIMIT={J1_LIMIT} Nm  v_max={V_MAX[0]}  a_max={A_MAX[0]}")
    print(f"KP={KP[0]}  KD={KD[0]}\n")

    tick = 0
    max_tau = 0.0
    max_tau_raw = 0.0
    max_ticks = 15000

    while tick < max_ticks:
        q_arm = np.array([data.qpos[a] for a in arm_qpos_adrs], dtype=float)
        qd_arm = np.array([data.qvel[a] for a in arm_dof_adrs], dtype=float)

        q_des, qd_des, qdd_des = traj.update()
        tau_arm, qdd_pd, qdd_cmd = ct.compute(model, data, q_des, qd_des, qdd_des, q_arm, qd_arm)

        tau_raw_j1 = tau_arm[0]
        # Hard saturation
        tau_arm[0] = J1_LIMIT * np.tanh(tau_arm[0] / J1_LIMIT)
        max_tau = max(max_tau, abs(tau_arm[0]))
        max_tau_raw = max(max_tau_raw, abs(tau_raw_j1))

        ctrl = np.zeros(model.nu, dtype=float)
        for i, jname in enumerate(ARM_JOINTS):
            ctrl[actuator_ids[jname]] = tau_arm[i]
        data.ctrl[:] = ctrl

        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        tick += 1
        if tick % 250 == 0:
            err = q_goal[0] - q_arm[0]
            trkErr = q_des[0] - q_arm[0]
            m_matrix = np.zeros((model.nv, model.nv), dtype=np.float64)
            mujoco.mj_fullM(model, m_matrix, data.qM)
            m00 = m_matrix[arm_dof_adrs[0], arm_dof_adrs[0]]
            bias0 = data.qfrc_bias[arm_dof_adrs[0]]
            print(f"  t={data.time:5.2f}s  J1={math.degrees(q_arm[0]):6.1f}  "
                  f"vel={qd_arm[0]:+5.2f}  "
                  f"tau={tau_arm[0]:+5.2f}  M00={m00:.4f}  bias={bias0:+.2f}")

        if abs(q_goal[0] - q_arm[0]) < math.radians(1.0) and abs(qd_arm[0]) < 0.05:
            print(f"\n  Done! t={data.time:.2f}s  max_tau={max_tau:.2f} Nm (raw: {max_tau_raw:.2f})")
            return

    print(f"\n  Timeout at t={data.time:.2f}s  J1={math.degrees(q_arm[0]):.1f}")


if __name__ == "__main__":
    main()

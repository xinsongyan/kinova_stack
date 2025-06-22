"""
Toggle the arm endlessly between the 'folded' and 'home' key-frames. Arrival condition and must hold for dwell_time seconds.
"""
from __future__ import annotations
import numpy as np
from kinova_env import SimEnv
from kinova_control.controllers import ArmPositionController


def arrived(q, qd, goal, tol_q, tol_qd, idx):
    """True if *both* position and velocity are within tolerance."""
    return np.all(np.abs(q[idx]  - goal) <= tol_q)


def main():
    # tolerances
    tol_q = 1
    tol_qd = 1
    dwell_time = 0.4 # seconds inside the band before switch

    # setup
    XML = "kinova_description/mjcf/m1n4s300_standalone.mjcf"
    sim = SimEnv(XML)
    sim.set_keyframe("folded")

    idx_arm = range(4) # joint slice
    q_folded = sim.get_keyframe_qpos("folded")[idx_arm]
    q_home = sim.get_keyframe_qpos("home")[idx_arm]

    arm_ctrl = ArmPositionController(kp=80, torque_limit=None, alpha=0.05)

    goal, name = q_home, "home"
    dwell_counter = 0
    dwell_steps = int(dwell_time / sim.dt)

    print("Toggling between 'folded' and 'home' … (Ctrl + C to quit)")

    # ------------------------------------------------------------- loop
    while sim._viewer is None or sim._viewer.is_running():
        q, qd = sim.state()

        # ---------- arrival detection BEFORE torque computation ----------
        if arrived(q, qd, goal, tol_q, tol_qd, idx_arm):
            dwell_counter += 1
            if dwell_counter >= dwell_steps:
                # switch target
                if name == "home":
                    goal, name = q_folded, "folded"
                else:
                    goal, name = q_home, "home"
                dwell_counter = 0
                print(f"-> now moving to '{name}'")
        else:
            # print current joint positions
            # print(f"Current joint positions are not close to target: {q[idx_arm]}")
            # print(f"Target joint positions: {goal}")

            # print(f"Current joint positions are not close to target: {np.abs(q[idx_arm] - goal).round(3)}")
            dwell_counter = 0

        # ---------- control torques --------------------------------------
        tau = np.zeros(sim.nu)
        tau[:4] = arm_ctrl(sim.data, goal)

        # print(f"Tau: {tau[:4].round(3)}")
        sim.set_torque(tau)
        sim.step()


if __name__ == "__main__":
    main()
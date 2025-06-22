"""
Run this script in the directory of the `kinova_env` package:
    python -m scripts.main
"""

import numpy as np
from kinova_env import SimEnv
from kinova_control.controllers import PDController, ArmPositionController, HandPositionController, OperationalSpaceController
from kinova_control.trajectory import interpolate
from kinova_control.gripper import finger_targets
import time

def main():
    sim = SimEnv("kinova_description/mjcf/m1n4s300_standalone.mjcf")
    sim.set_keyframe("folded")

    
    # ---------- gains
    # 4 arm joints, 3 proximal fingers, 3 distal fingers
    Kp = np.array([120, 120, 120, 120, 1, 1, 1, 1, 1, 1])

    # Kp = np.array([100, 100, 100, 100, 1, 1, 1, 1, 1, 1])
    Kd = np.array([1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    pd_controller = PDController(Kp, Kd)

    # ---------- trajectory
    q0 = sim.get_keyframe_qpos("folded")
    q1 = sim.get_keyframe_qpos("home")
    T  = 2.0 # [s] fold to home

    DZ_TOTAL = 0.10 # metres (10 cm)
    DZ_STEP = 0.02 # metres per incremental move
    POS_TOL = 2e-3 # = 2 mm tolerance for ee_at_position

    pos0, _  = sim.ee_pose() # current EE position
    goal_pos = pos0 + np.array([0.0, 0.0, DZ_TOTAL])

    print(f"Current EE position: {pos0.round(3)} [m]")
    print(f"Target  EE position: {goal_pos.round(3)} [m] (+{DZ_TOTAL*100:.0f} cm)")
    

    # ---------- main loop ----------
    
    while True:
        t = sim.time()

        q, qd = sim.state()
        q_des = interpolate(q0, q1, t, T)
        tau = pd_controller(q, qd, q_des)

        # if sim.ee_at_position(goal_pos, tol=POS_TOL):
        #     print("Reached target position!")

        flag, delta = sim.joints_close(q1, tol=0.02, indices=range(4))
        if flag:
            print("Reached target joint positions!")
            # sim.print_ee_pose()
            # Should be: goal_pos = [0.077, -0.236,  0.559]

        else:
            print(f"Current joint positions are not close to target: {delta.round(3)}")
            
        sim.set_torque(tau)
        sim.step()
    

if __name__ == "__main__":
    main()

import os
import numpy as np

from sim_env import SimEnv
from controller import PDController



def main():
    sim = SimEnv("kinova_description/mjcf/m1n4s300_standalone.mjcf")
    
    Kp = np.array([100, 100, 100, 100, 1, 1, 1, 1, 1, 1])  # Proportional gains
    Kd = np.array([1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # Derivative gains
    pd_controller = PDController(Kp=Kp, Kd=Kd)
    
    q_desired = sim.get_model_keyframe("home1").qpos  # shape (model.nu,)

    while True:
        q, qd = sim.get_state()
        tau = pd_controller.compute(q, qd, q_desired)
        sim.set_ctrl(tau)
        sim.step()


if __name__ == "__main__":
    main()
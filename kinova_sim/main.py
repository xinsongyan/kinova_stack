
import os
import numpy as np

from sim_env import SimEnv
from controller import PDController



def main():
    sim = SimEnv("kinova_description/mjcf/m1n4s300_standalone.mjcf")
    sim.set_model_keyframe("folded")  # Set the initial keyframe

    Kp = np.array([100, 100, 100, 100, 1, 1, 1, 1, 1, 1])  # Proportional gains
    Kd = np.array([1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # Derivative gains
    pd_controller = PDController(Kp=Kp, Kd=Kd)
    
    q_ini = sim.get_model_keyframe("folded").qpos  # shape (model.nq,)
    q_des = sim.get_model_keyframe("home").qpos  # shape (model.nu,)
    T = 2.0  # Duration for the transition from q_ini to q_des
    while True:
        t =  sim.get_time()
        q, qd = sim.get_state()
        
        # Interpolate q_desired from q_ini to q_des over T=2 seconds
        alpha = min(t / T, 1.0)
        q_cmd = (1 - alpha) * q_ini + alpha * q_des
        

        tau = pd_controller.compute(q, qd, q_cmd)
        sim.set_ctrl(tau)
        sim.step()
  


if __name__ == "__main__":
    main()
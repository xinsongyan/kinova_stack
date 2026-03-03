"""Test: why doesn't J1 accelerate when tau=2.8 Nm?"""
import os
import mujoco
import numpy as np
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF = os.path.realpath(os.path.join(_HERE, "kinova_description", "mjcf", "m1n4s300_standalone.mjcf"))
_MESH = os.path.realpath(os.path.join(_HERE, "kinova_description", "meshes"))

with open(_MJCF, "r") as f:
    xml = f.read().replace('<compiler ', f'<compiler meshdir="{_MESH}" ', 1)

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

HOME_QPOS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
data.qpos[:10] = HOME_QPOS
# Fast forward J1 to 380 degrees (6.63 rad)
data.qpos[0] = math.radians(380.0)

mujoco.mj_forward(model, data)

act_j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_joint_1")

# Apply 3.0 Nm
print("Applying 3.0 Nm to J1 (at 380 deg), holding other joints zero...")
for step in range(100):
    # Hold J2/J3/J4 at HOME_QPOS using very strong PD
    for i in range(1, 4):
        err = HOME_QPOS[i] - data.qpos[i]
        err_d = -data.qvel[i]
        # Strong holding torque
        qdd = 1000.0 * err + 200.0 * err_d
        tau = 0.1 * qdd # approx M=0.1
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_joint_{i+1}")
        data.ctrl[act_id] = tau

    # Apply exactly 3.0 Nm to J1
    data.ctrl[act_j1] = 3.0
    
    mujoco.mj_step(model, data)
    
    if step % 10 == 0:
        print(f"step {step:3d}: J1 vel = {data.qvel[0]:.4f} rad/s, J1 pos = {math.degrees(data.qpos[0]):.2f} deg")

print("Finished 100 steps.")

import mujoco
import mujoco.viewer
import os



cwd = os.getcwd()
print(f"Current working directory: {cwd}")
os.chdir(cwd)


model = mujoco.MjModel.from_xml_path("mico.urdf")
data = mujoco.MjData(model)



# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

import mujoco
import mujoco.viewer
import os



script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current script directory: {script_dir}")
project_dir = os.path.dirname(os.path.dirname(script_dir))
print(f"Project directory: {project_dir}")
os.chdir(project_dir)


model = mujoco.MjModel.from_xml_path("kinova_description/urdf/m1n4s300_standalone.urdf")
data = mujoco.MjData(model)

output_dir = "kinova_description/mjcf"
output_path = os.path.join(output_dir, "m1n4s300_standalone.mjcf")
mujoco.mj_saveLastXML(output_path, model)
print(f"Model saved to {output_path}")

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

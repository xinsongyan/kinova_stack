import mujoco
import mujoco.viewer
import os


def model_from_urdf(urdf_path):
    """Load a Mujoco model from a URDF file."""
    return mujoco.MjModel.from_xml_path(urdf_path)

def model_from_mjcf(mjcf_path):
    """Load a Mujoco model from a MJCF file."""
    return mujoco.MjModel.from_xml_path(mjcf_path)









def main():
    
    # model = model_from_urdf("kinova_description/urdf/m1n4s300_standalone.urdf")
    model = model_from_mjcf("kinova_description/mjcf/m1n4s300_standalone.mjcf")
    data = mujoco.MjData(model)


    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Press ESC to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
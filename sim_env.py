import mujoco
import mujoco.viewer
import numpy as np
import sys


def get_test_model_path():
    import importlib.resources
    with importlib.resources.path("mujoco", "testdata") as model_dir:
        model_path = model_dir / "model.xml"
        return str(model_path)



class SimEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.nu = self.model.nu

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        



    def step(self):
        mujoco.mj_step(self.model, self.data)
        self.sync_viewer()
        
    def sync_viewer(self):
        if self.viewer.is_running():
            self.viewer.sync()
        else:
            sys.exit("Viewer closed. Exiting simulation.")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_model_keyframe(self, name):
        keyframe = self.model.keyframe(name=name)
        if keyframe is None:
            raise ValueError(f"Keyframe '{name}' not found in the model.")
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe.id)


    def get_state(self):
        return np.copy(self.data.qpos), np.copy(self.data.qvel)

    def get_time(self):
        return self.data.time

    def get_model_keyframe(self, name):
        return self.model.keyframe(name=name)

    def set_ctrl(self, ctrl):
        self.data.ctrl[:] = ctrl


if __name__ == "__main__":
    # Example usage
    model_path = get_test_model_path()  
    env = SimEnv(model_path)

    while True:
        env.step()
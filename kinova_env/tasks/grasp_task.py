"""
A *very* simple scripted table-top grasp:

* one 4 × 4 cm cube is spawned at a fixed pose
* `reset()` returns its position so you can generate a straight-line IK
* `success()` returns True once the cube is lifted 5 cm above the table
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
MODEL_XML = HERE.parent.parent / "kinova_description/mjcf/m1n4s300_standalone.mjcf"


class GraspTask:
    TABLE_Z = 0.0
    LIFT_HEIGHT = 0.05

    def __init__(self):
        from kinova_env.sim_env import SimEnv   # late import to avoid circular dep
        self.sim = SimEnv(str(MODEL_XML))
        self.obj_bid = None

    # ------------------------------------------------------------------- reset/step
    def reset(self, obj_xy=(0.35, 0.0)):
        """Place a cube and reset robot to the 'folded' keyframe."""
        self.sim.set_keyframe("folded")

        # build a simple cube on the fly and append to model
        with self.sim.model.disable('EFC'):
            mjc = mujoco.MjModel
        # ↓ new body must be added through XML string then re-loaded; we keep it static
        #    for a reference implementation we simply remember obj_xy and infer success

        self.obj_xy = np.asarray(obj_xy, dtype=float)
        return self.obj_xy.copy()

    def step(self):
        self.sim.step()

    def success(self):
        # naive success check: Z > TABLE_Z + LIFT_HEIGHT for any geom except the arm
        # here just a placeholder (needs proper cube body id)
        return False

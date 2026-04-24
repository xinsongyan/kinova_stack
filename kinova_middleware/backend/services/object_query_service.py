from __future__ import annotations

import mujoco
import numpy as np

from kinova_middleware.backend.runtime.mujoco_runtime import MuJoCoRuntimeAdapter


class MuJoCoObjectQueryService:
    """Own scene object pose/metadata lookup for the MuJoCo runtime."""

    def __init__(self, runtime: MuJoCoRuntimeAdapter) -> None:
        self._runtime = runtime

    def get_object_pose(self, body_name: str) -> dict:
        model = self._runtime.model
        data = self._runtime.data

        geom_type_names = {
            0: "plane",
            1: "hfield",
            2: "sphere",
            3: "capsule",
            4: "ellipsoid",
            5: "cylinder",
            6: "box",
            7: "mesh",
        }

        def format_body_pose(body_id: int) -> dict:
            body_name_resolved = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            pos = data.xpos[body_id].copy()
            quat_wxyz = data.xquat[body_id].copy()

            geom_size: list[float] = []
            geom_type_str = "unknown"
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] != body_id:
                    continue
                geom_type_int = int(model.geom_type[geom_id])
                geom_type_str = geom_type_names.get(geom_type_int, f"type_{geom_type_int}")
                raw_size = model.geom_size[geom_id].copy()
                if geom_type_str == "cylinder":
                    geom_size = [float(raw_size[0]), float(raw_size[1])]
                elif geom_type_str == "box":
                    geom_size = [float(raw_size[0]), float(raw_size[1]), float(raw_size[2])]
                elif geom_type_str == "sphere":
                    geom_size = [float(raw_size[0])]
                else:
                    geom_size = [float(v) for v in raw_size]
                break

            return {
                "body_name": body_name_resolved,
                "position": {
                    "x": round(float(pos[0]), 4),
                    "y": round(float(pos[1]), 4),
                    "z": round(float(pos[2]), 4),
                },
                "size": [round(s, 4) for s in geom_size],
                "geom_type": geom_type_str,
                "quaternion": {
                    "qx": round(float(quat_wxyz[1]), 4),
                    "qy": round(float(quat_wxyz[2]), 4),
                    "qz": round(float(quat_wxyz[3]), 4),
                    "qw": round(float(quat_wxyz[0]), 4),
                },
            }

        if body_name == "all":
            objects = []
            for body_id in range(model.nbody):
                joint_adr = model.body_jntadr[body_id]
                joint_num = model.body_jntnum[body_id]
                if joint_num > 0 and model.jnt_type[joint_adr] == mujoco.mjtJoint.mjJNT_FREE:
                    objects.append(format_body_pose(body_id))
            return {"status": "ok", "objects": objects}

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            return format_body_pose(body_id)

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, body_name)
        if site_id < 0:
            return {"status": "error", "message": f"Body or site '{body_name}' not found in model."}

        pos = data.site(site_id).xpos.copy()
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        site_view = data.site(site_id)
        if hasattr(site_view, "xquat"):
            quat_wxyz = site_view.xquat.copy()

        return {
            "body_name": body_name,
            "position": {
                "x": round(float(pos[0]), 4),
                "y": round(float(pos[1]), 4),
                "z": round(float(pos[2]), 4),
            },
            "size": [0.01],
            "geom_type": "site",
            "quaternion": {
                "qx": round(float(quat_wxyz[1]), 4),
                "qy": round(float(quat_wxyz[2]), 4),
                "qz": round(float(quat_wxyz[3]), 4),
                "qw": round(float(quat_wxyz[0]), 4),
            },
        }

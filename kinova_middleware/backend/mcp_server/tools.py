import logging
import math
import ast
from typing import Any
from fastmcp import FastMCP

log = logging.getLogger("mcp_kinova")

def _coerce_to_float_list(value: Any, name: str) -> list[float]:
    """Coerce incoming value to a list of floats.

    Accepts actual lists/tuples of numbers or string-encoded lists like
    "[0.03, 0.09]". Raises ValueError when conversion fails.
    """
    if isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except Exception as exc:
            raise ValueError(f"{name} must contain numeric elements ({exc})") from exc

    if isinstance(value, str):
        # Try safe literal eval first (handles '[1, 2]' and '(1,2)')
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = None

        if isinstance(parsed, (list, tuple)):
            try:
                return [float(v) for v in parsed]
            except Exception as exc:
                raise ValueError(f"{name} must contain numeric elements ({exc})") from exc

        # Fallback: split on commas for simple CSV-like strings
        try:
            parts = [p.strip() for p in value.strip().strip('()[]').split(',') if p.strip()]
            return [float(p) for p in parts]
        except Exception as exc:
            raise ValueError(f"Could not parse {name} from string: {exc}") from exc

    raise ValueError(f"{name} must be a list or string-encoded list; got {type(value).__name__}")

def setup_tools(mcp: FastMCP, state: dict):
    """Register Kinova tools with the MCP server."""
    get_controller = state["get_controller"]
    motion_lock = state["motion_lock"]
    physics_lock = state["physics_lock"]
    run_until_reached = state["run_until_reached"]

    def _quat_rotate(q: list[float], v: list[float]) -> list[float]:
        """Rotate vector v by quaternion q = (qx, qy, qz, qw) using q * v * q^-1."""
        qx, qy, qz, qw = q
        t = [
            2.0 * (qy * v[2] - qz * v[1]),
            2.0 * (qz * v[0] - qx * v[2]),
            2.0 * (qx * v[1] - qy * v[0]),
        ]
        return [
            v[0] + qw * t[0] + qy * t[2] - qz * t[1],
            v[1] + qw * t[1] + qz * t[0] - qx * t[2],
            v[2] + qw * t[2] + qx * t[1] - qy * t[0],
        ]

    def _quat_rotation_error(q1: tuple[float, ...], q2: tuple[float, ...]) -> float:
        """Rotation error magnitude (rad) between two unit quaternions (xyzw)."""
        dot = abs(sum(a * b for a, b in zip(q1, q2)))
        dot = min(dot, 1.0)
        return 2.0 * math.acos(dot)

    # ...continued inside setup_tools...

    # ── Tool 0: reset_scene ──────────────────────────────────────────────────
    @mcp.tool()
    def reset_scene() -> dict:
        """Reset the simulation physics, returning the environment to its initial state.

        Returns:
            status: "ok" or "error"
            message: human-readable result
        """
        log.info("Tool  reset_scene()")
        ctrl = get_controller()

        with motion_lock:
            with physics_lock:
                try:
                    ctrl.reset_scene()
                except Exception as e:
                    log.error("reset_scene failed: %s", e)
                    return {"status": "error", "message": f"Reset failed: {e}"}

        log.info("  → Scene reset successfully.")
        return {"status": "ok", "message": "Scene reset successfully."}

    # ── Tool 1: move_home ──────────────────────────────────────────────────────
    @mcp.tool()
    def move_home() -> dict:
        """Command the arm to its home configuration and block until reached.

        Returns:
            status: "ok" or "timeout"
            message: human-readable result
        """
        log.info("Tool  move_home()")
        ctrl = get_controller()
        with motion_lock:
            ctrl.move_home()
            reached = run_until_reached()
        status = "ok" if reached else "timeout"
        msg = "Home reached." if reached else "Timed out moving home."
        log.info("  → %s", msg)
        return {"status": status, "message": msg}

    # ── Tool 2: get_end_effector_pose ──────────────────────────────────────────
    @mcp.tool()
    def get_end_effector_pose() -> dict:
        """Read the current end-effector Cartesian pose (non-blocking).

        Returns:
            position: {x, y, z}  (metres)
            quaternion: {qx, qy, qz, qw}
        """
        ctrl = get_controller()
        with physics_lock:
            pos, quat = ctrl.get_end_effector_pose()
        return {
            "position": {"x": round(pos[0], 4), "y": round(pos[1], 4), "z": round(pos[2], 4)},
            "quaternion": {"qx": round(quat[0], 4), "qy": round(quat[1], 4), "qz": round(quat[2], 4), "qw": round(quat[3], 4)},
        }

    # ── Tool 4: set_gripper ──────────────────────────────────────────────────
    @mcp.tool()
    def set_gripper(percent: float) -> dict:
        """Set gripper opening. 1.0 = fully open, 0.0 = fully closed.

        Blocks briefly to let the gripper settle.  Fingers are torque-limited
        (forcerange in MJCF), so they will stop closing when contact force
        reaches the cap.

        Args:
            percent: opening ratio in [0.0, 1.0]

        Returns:
            status: "ok"
            percent: actual commanded percent
            contact_detected: True if fingers hit something
            max_actuator_force: peak finger force (N)
            message: description
        """
        log.info("Tool  set_gripper(%.2f)", percent)
        ctrl = get_controller()
        p = max(0.0, min(1.0, float(percent)))
        with motion_lock:
            current_percent = None
            try:
                state = ctrl.get_gripper_state()
                current_percent = state.get("percent")
            except Exception:
                current_percent = None

            commands = [p]
            if current_percent is not None and p > current_percent + 0.15:
                n_steps = max(2, min(5, math.ceil((p - current_percent) / 0.12)))
                commands = [
                    current_percent + (p - current_percent) * ((i + 1) / n_steps)
                    for i in range(n_steps)
                ]

            for i, cmd_percent in enumerate(commands):
                ctrl.set_gripper_percent(cmd_percent)
                final_step = i == len(commands) - 1
                ctrl.wait_for_gripper(
                    timeout_s=5.0 if final_step else 1.0,
                    hold_seconds=0.2 if final_step else 0.05,
                    hz=500.0,
                    pos_tol_rad=0.05,
                    vel_tol_rad_s=0.2,
                )

        try:
            force_info = ctrl.get_finger_forces()
            max_force = force_info.get("max_abs_force", 0.0)
        except Exception:
            max_force = 0.0

        msg = f"Gripper set to {p*100:.0f}%."
        log.info("  → %s", msg)
        return {
            "status": "ok",
            "percent": round(p, 4),
            "max_actuator_force": round(max_force, 4),
            "message": msg,
        }

    # ── Tool 6: move_pose ────────────────────────────────────────────────────
    @mcp.tool()
    def move_pose(
        target_pos: list[float] | str,
        target_quat: list[float] | str,
        seed_q_rad: list[float] | str | None = None,
        allow_orientation_fallback: bool = True,
        move_wrist: bool = True,
    ) -> dict:
        """Move the end-effector to a Cartesian pose (IK → joint command → block).

        Args:
            target_pos: [x, y, z] in metres, or a string-encoded list
            target_quat: [qx, qy, qz, qw] unit quaternion, or a string-encoded list
            seed_q_rad: optional IK seed (arm joints)
            allow_orientation_fallback: if True and quaternion is invalid,
                fall back to position-only IK
            move_wrist: if False, the wrist joint (last arm joint) is kept frozen
                at its current position during IK solving.

        Returns:
            status: "ok" | "timeout" | "error"
            mode_used: "full_pose" | "position_only"
            q_target_rad: IK solution sent to arm
            final_pose: {position, quaternion}
            pos_err: Euclidean position error (m)
            rot_err: rotation error (rad), null if position-only
        """
        ctrl = get_controller()
        try:
            target_pos = _coerce_to_float_list(target_pos, "target_pos")
            target_quat = _coerce_to_float_list(target_quat, "target_quat")
            if seed_q_rad is not None:
                seed_q_rad = _coerce_to_float_list(seed_q_rad, "seed_q_rad")
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        log.info("Tool  move_pose(pos=%s, quat=%s, move_wrist=%s)", target_pos, target_quat, move_wrist)

        if len(target_pos) != 3:
            return {"status": "error", "message": "target_pos must have 3 elements."}
        if len(target_quat) != 4:
            return {"status": "error", "message": "target_quat must have 4 elements."}

        quat_norm = math.sqrt(sum(v * v for v in target_quat))
        mode_used = "full_pose"

        if quat_norm < 1e-6:
            if allow_orientation_fallback:
                log.warning("  Quaternion near-zero → position-only fallback")
                with physics_lock:
                    _, cur_quat = ctrl.get_end_effector_pose()
                target_quat = list(cur_quat)
                mode_used = "position_only"
            else:
                return {"status": "error", "message": "Invalid quaternion (near-zero norm)."}
        elif not move_wrist:
            log.warning("  move_wrist=False → forcing position-only IK fallback to preserve wrist angle")
            with physics_lock:
                _, cur_quat = ctrl.get_end_effector_pose()
            mode_used = "position_only"
        else:
            target_quat = [v / quat_norm for v in target_quat]

        try:
            with physics_lock:
                if mode_used == "position_only":
                    q_target = ctrl.solve_ik_position_only(target_pos, seed_q_rad, move_wrist=move_wrist)
                else:
                    q_target = ctrl.solve_ik(target_pos, target_quat, seed_q_rad, move_wrist=move_wrist)
        except ValueError as exc:
            log.error("  IK / safety error: %s", exc)
            return {"status": "error", "message": f"ik_failed: {exc}"}

        with motion_lock:
            ctrl.send_joint_position_rad(q_target)
            reached = run_until_reached()

        with physics_lock:
            final_pos, final_quat = ctrl.get_end_effector_pose()

        pos_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(final_pos, target_pos)))
        
        rot_err = None
        if quat_norm >= 1e-6:
            rot_err = _quat_rotation_error(final_quat, tuple(target_quat))

        if pos_err > 0.04:
            status = "error"
            msg = f"ik_failed: position error too large (err={pos_err:.4f}m)"
            log.error("  %s", msg)
            return {"status": status, "message": msg, "mode_used": mode_used, "pos_err": pos_err, "rot_err": rot_err}
        
        # if rot_err is not None:
        #     if not move_wrist and rot_err > 0.3:
        #         status = "error"
        #         msg = f"ik_failed: orientation infeasible with move_wrist=False (err={rot_err:.4f}rad)"
        #         log.error("  %s", msg)
        #         return {"status": status, "message": msg, "mode_used": mode_used, "pos_err": pos_err, "rot_err": rot_err}
        #     elif move_wrist and rot_err > 0.2:
        #         status = "error"
        #         msg = f"ik_failed: orientation error too large (err={rot_err:.4f}rad)"
        #         log.error("  %s", msg)
        #         return {"status": status, "message": msg, "mode_used": mode_used, "pos_err": pos_err, "rot_err": rot_err}

        status = "ok" if reached else "timeout"
        log.info("  → %s  pos_err=%.4f m  rot_err=%s rad", status, pos_err, f"{rot_err:.4f}" if rot_err is not None else "n/a")
        return {
            "status": status,
            "mode_used": mode_used,
            "q_target_rad": [round(q, 4) for q in q_target],
            "final_pose": {
                "position": [round(p, 4) for p in final_pos],
                "quaternion": [round(q, 4) for q in final_quat]
            },
            "pos_err": round(pos_err, 4),
            "rot_err": round(rot_err, 4) if rot_err is not None else None,
        }

    # ── Tool 7: rotate_wrist ──────────────────────────────────────────────────
    @mcp.tool()
    def rotate_wrist(angle_deg: float) -> dict:
        """Rotate the wrist (last arm joint) by a relative angle.

        Args:
            angle_deg: rotation angle in degrees (positive or negative)

        Returns:
            status: "ok" | "error"
            message: description of result
        """
        ctrl = get_controller()
        log.info("Tool  rotate_wrist(%.1f deg)", angle_deg)
        
        with motion_lock:
            try:
                ctrl.rotate_wrist(angle_deg)
                reached = run_until_reached()
                status = "ok" if reached else "timeout"
            except Exception as e:
                log.error("rotate_wrist failed: %s", e)
                return {"status": "error", "message": str(e)}

        return {"status": status, "message": f"Rotate wrist by {angle_deg} deg finished. Reached={reached}"}

    # ── Tool 8: get_object_pose ──────────────────────────────────────────────
    @mcp.tool()
    def get_object_pose(body_name: str) -> dict:
        """Read the Cartesian pose of any named body in the MuJoCo scene.

        Useful for locating objects like the cube, table, or any other body.
        If you use body_name = "all", it will return the pose of all bodies.

        Args:
            body_name: name of the body in the MJCF model (e.g. "cube")

        Returns:
            body_name: the queried body name
            position: {x, y, z}  (metres)
            size: list of geom size parameters
            geom_type: string name of the geom type
            quaternion: {qx, qy, qz, qw}
        """
        ctrl = get_controller()
        log.info("Tool  get_object_pose(%s)", body_name)

        backend = getattr(ctrl, "_backend", None)
        inner = getattr(backend, "_inner", backend)
        env = getattr(inner, "_env", None)
        if env is None:
            return {"status": "error", "message": "Cannot access MuJoCo environment."}

        import mujoco as _mj
        model = env.model
        data = env.data

        def _get_body_info(bid):
            bname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_BODY, bid)
            
            with physics_lock:
                pos = data.xpos[bid].copy()
                quat_wxyz = data.xquat[bid].copy()

            _GEOM_TYPE_NAMES = {
                0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
                4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh",
            }
            geom_size = []
            geom_type_str = "unknown"
            
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid:
                    geom_type_int = int(model.geom_type[gid])
                    geom_type_str = _GEOM_TYPE_NAMES.get(geom_type_int, f"type_{geom_type_int}")
                    raw_size = model.geom_size[gid].copy()
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
                "body_name": bname,
                "position": {"x": round(float(pos[0]), 4), "y": round(float(pos[1]), 4), "z": round(float(pos[2]), 4)},
                "size": [round(s, 4) for s in geom_size],
                "geom_type": geom_type_str,
                "quaternion": {
                    "qx": round(float(quat_wxyz[1]), 4), "qy": round(float(quat_wxyz[2]), 4), 
                    "qz": round(float(quat_wxyz[3]), 4), "qw": round(float(quat_wxyz[0]), 4)
                }
            }

        if body_name == "all":
            objects = []
            for bid in range(model.nbody):
                jnt_adr = model.body_jntadr[bid]
                jnt_num = model.body_jntnum[bid]
                if jnt_num > 0 and model.jnt_type[jnt_adr] == _mj.mjtJoint.mjJNT_FREE:
                    objects.append(_get_body_info(bid))
            return {"status": "ok", "objects": objects}

        body_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            site_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_SITE, body_name)
            if site_id < 0:
                return {"status": "error", "message": f"Body or site '{body_name}' not found in model."}
            
            with physics_lock:
                pos = data.site(site_id).xpos.copy()
                quat_wxyz = [1.0, 0.0, 0.0, 0.0]
                if hasattr(data.site(site_id), 'xquat'):
                    quat_wxyz = data.site(site_id).xquat.copy()
                
            return {
                "body_name": body_name,
                "position": {"x": round(float(pos[0]), 4), "y": round(float(pos[1]), 4), "z": round(float(pos[2]), 4)},
                "size": [0.01],
                "geom_type": "site",
                "quaternion": {
                    "qx": round(float(quat_wxyz[1]), 4), "qy": round(float(quat_wxyz[2]), 4), 
                    "qz": round(float(quat_wxyz[3]), 4), "qw": round(float(quat_wxyz[0]), 4)
                }
            }

        return _get_body_info(body_id)

    @mcp.tool()
    def compute_grasp_height(geom_type: str, size: list[float] | str, quat_xyzw: list[float] | str) -> dict:
        """Compute the height of the object's top surface above its body origin.

        This tool accepts `size` and `quat_xyzw` either as native lists/tuples of
        numbers or as string-encoded lists like "[0.03, 0.09]". It will coerce
        string inputs to real lists and return a helpful error message if
        coercion fails.
        """
        valid_types = ["cylinder", "box", "sphere"]
        if geom_type not in valid_types:
            return {
                "status": "error",
                "message": f"Invalid geom_type '{geom_type}'. Accepted types: {', '.join(valid_types)}",
                "top_height": 0.0,
            }

        try:
            size_list = _coerce_to_float_list(size, "size")
        except ValueError as exc:
            return {"status": "error", "message": str(exc), "top_height": 0.0}

        try:
            quat_list = _coerce_to_float_list(quat_xyzw, "quat_xyzw")
        except ValueError as exc:
            return {"status": "error", "message": str(exc), "top_height": 0.0}

        if len(quat_list) != 4:
            return {"status": "error", "message": "quat_xyzw must have 4 elements", "top_height": 0.0}

        qx, qy, qz, qw = quat_list
        top_z = 0.0

        if geom_type == "cylinder":
            if len(size_list) < 2:
                return {"status": "error", "message": "Cylinder size must be [radius, half_height]", "top_height": 0.0}
            radius, half_height = float(size_list[0]), float(size_list[1])
            axis_world = _quat_rotate([qx, qy, qz, qw], [0, 0, half_height])
            rx = _quat_rotate([qx, qy, qz, qw], [radius, 0, 0])
            ry = _quat_rotate([qx, qy, qz, qw], [0, radius, 0])
            top_z = abs(axis_world[2]) + max(abs(rx[2]), abs(ry[2]))

        elif geom_type == "box":
            if len(size_list) < 3:
                return {"status": "error", "message": "Box size must be [hx, hy, hz]", "top_height": 0.0}
            hx, hy, hz = float(size_list[0]), float(size_list[1]), float(size_list[2])
            vx = _quat_rotate([qx, qy, qz, qw], [hx, 0, 0])
            vy = _quat_rotate([qx, qy, qz, qw], [0, hy, 0])
            vz = _quat_rotate([qx, qy, qz, qw], [0, 0, hz])
            top_z = abs(vx[2]) + abs(vy[2]) + abs(vz[2])

        elif geom_type == "sphere":
            if len(size_list) < 1:
                return {"status": "error", "message": "Sphere size must be [radius]", "top_height": 0.0}
            top_z = float(size_list[0])

        return {"status": "ok", "top_height": round(top_z + 0.03, 4)}

    @mcp.tool()
    def compute_wrist_alignment(obj_quat_xyzw: list[float] | str, ee_quat_xyzw: list[float] | str) -> dict:
        """Compute the wrist rotation needed to align the EE X-axis with the object's long axis.

        Accepts quaternion inputs as lists/tuples or string-encoded lists; coerces them
        and returns a helpful error on failure.
        """
        try:
            obj_q = _coerce_to_float_list(obj_quat_xyzw, "obj_quat_xyzw")
            ee_q = _coerce_to_float_list(ee_quat_xyzw, "ee_quat_xyzw")
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        if len(obj_q) != 4 or len(ee_q) != 4:
            return {"status": "error", "message": "Quaternions must have 4 elements"}

        cyl_axis = _quat_rotate(obj_q, [0, 0, 1])
        cyl_angle = math.atan2(cyl_axis[1], cyl_axis[0])
        ee_x = _quat_rotate(ee_q, [1, 0, 0])
        ee_x_angle = math.atan2(ee_x[1], ee_x[0])
        diff_rad = cyl_angle - ee_x_angle
        diff_rad = (diff_rad + math.pi) % (2 * math.pi) - math.pi
        return {"status": "ok", "angle_deg": round(math.degrees(diff_rad), 4)}

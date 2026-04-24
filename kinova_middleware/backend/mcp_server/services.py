from __future__ import annotations

import ast
from dataclasses import dataclass
import logging
import math
from typing import Any, Callable


WORKSPACE_MIN_RADIUS_M = 0.10
WORKSPACE_MAX_RADIUS_M = 0.50


def _coerce_to_float_list(value: Any, name: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except Exception as exc:
            raise ValueError(f"{name} must contain numeric elements ({exc})") from exc

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = None

        if isinstance(parsed, (list, tuple)):
            try:
                return [float(v) for v in parsed]
            except Exception as exc:
                raise ValueError(f"{name} must contain numeric elements ({exc})") from exc

        try:
            parts = [p.strip() for p in value.strip().strip("()[]").split(",") if p.strip()]
            return [float(p) for p in parts]
        except Exception as exc:
            raise ValueError(f"Could not parse {name} from string: {exc}") from exc

    raise ValueError(f"{name} must be a list or string-encoded list; got {type(value).__name__}")


def _validate_workspace_target(target_pos: list[float]) -> None:
    if len(target_pos) != 3:
        raise ValueError("target_pos must have 3 elements.")

    if not all(math.isfinite(v) for v in target_pos):
        raise ValueError("target_pos must contain only finite numeric values.")

    radius = math.hypot(target_pos[0], target_pos[1])
    if radius < WORKSPACE_MIN_RADIUS_M or radius > WORKSPACE_MAX_RADIUS_M:
        raise ValueError(
            "target_pos is outside the workspace: "
            f"xy radius={radius:.3f} m, allowed range="
            f"[{WORKSPACE_MIN_RADIUS_M:.2f}, {WORKSPACE_MAX_RADIUS_M:.2f}] m."
        )


def _quat_rotate(q: list[float], v: list[float]) -> list[float]:
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
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = min(dot, 1.0)
    return 2.0 * math.acos(dot)


@dataclass(slots=True)
class ToolRuntimeContext:
    get_controller: Callable[[], Any]
    motion_lock: Any
    physics_lock: Any
    run_until_reached: Callable[..., bool]
    reset_or_reload_scene: Callable[[int | None], dict] | None = None


class KinovaMotionToolService:
    def __init__(
        self,
        runtime: ToolRuntimeContext,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._runtime = runtime
        self._log = logger or logging.getLogger("mcp_kinova")

    def reset_scene(self, scene_number: int | str | None = None) -> dict:
        self._log.info("Tool  reset_scene(scene_number=%r)", scene_number)
        parsed_scene_number: int | None = None
        if scene_number is not None:
            try:
                parsed_scene_number = int(scene_number)
            except (TypeError, ValueError) as exc:
                return {
                    "status": "error",
                    "message": f"scene_number must be an integer when provided ({exc}).",
                }

        with self._runtime.motion_lock:
            try:
                if self._runtime.reset_or_reload_scene is not None:
                    result = self._runtime.reset_or_reload_scene(parsed_scene_number)
                else:
                    with self._runtime.physics_lock:
                        ctrl = self._runtime.get_controller()
                        ctrl.reset_scene()
                        result = {
                            "status": "ok",
                            "message": "Scene reset successfully.",
                            "scene_changed": False,
                        }
            except Exception as exc:
                self._log.error("reset_scene failed: %s", exc)
                return {"status": "error", "message": f"Reset failed: {exc}"}

        self._log.info("  → %s", result.get("message", "Scene reset successfully."))
        return result

    def move_home(self) -> dict:
        self._log.info("Tool  move_home()")
        ctrl = self._runtime.get_controller()
        with self._runtime.motion_lock:
            ctrl.move_home()
            reached = self._runtime.run_until_reached()
        status = "ok" if reached else "timeout"
        msg = "Home reached." if reached else "Timed out moving home."
        self._log.info("  → %s", msg)
        return {"status": status, "message": msg}

    def get_end_effector_pose(self) -> dict:
        ctrl = self._runtime.get_controller()
        with self._runtime.physics_lock:
            pos, quat = ctrl.get_end_effector_pose()
        return {
            "position": {"x": round(pos[0], 4), "y": round(pos[1], 4), "z": round(pos[2], 4)},
            "quaternion": {
                "qx": round(quat[0], 4),
                "qy": round(quat[1], 4),
                "qz": round(quat[2], 4),
                "qw": round(quat[3], 4),
            },
        }

    def set_gripper(self, percent: float) -> dict:
        self._log.info("Tool  set_gripper(%.2f)", percent)
        ctrl = self._runtime.get_controller()
        opening = max(0.0, min(1.0, float(percent)))
        with self._runtime.motion_lock:
            current_percent = None
            try:
                state = ctrl.get_gripper_state()
                current_percent = state.get("percent")
            except Exception:
                current_percent = None

            commands = [opening]
            if current_percent is not None and opening > current_percent + 0.15:
                n_steps = max(2, min(5, math.ceil((opening - current_percent) / 0.12)))
                commands = [
                    current_percent + (opening - current_percent) * ((i + 1) / n_steps)
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

        msg = f"Gripper set to {opening*100:.0f}%."
        self._log.info("  → %s", msg)
        return {
            "status": "ok",
            "percent": round(opening, 4),
            "max_actuator_force": round(max_force, 4),
            "message": msg,
        }

    def move_pose(
        self,
        target_pos: list[float] | str,
        target_quat: list[float] | str,
        seed_q_rad: list[float] | str | None = None,
        *,
        allow_orientation_fallback: bool = True,
        move_wrist: bool = True,
    ) -> dict:
        ctrl = self._runtime.get_controller()
        try:
            target_pos = _coerce_to_float_list(target_pos, "target_pos")
            target_quat = _coerce_to_float_list(target_quat, "target_quat")
            if seed_q_rad is not None:
                seed_q_rad = _coerce_to_float_list(seed_q_rad, "seed_q_rad")
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        self._log.info("Tool  move_pose(pos=%s, quat=%s, move_wrist=%s)", target_pos, target_quat, move_wrist)

        if len(target_pos) != 3:
            return {"status": "error", "message": "target_pos must have 3 elements."}
        if len(target_quat) != 4:
            return {"status": "error", "message": "target_quat must have 4 elements."}
        try:
            _validate_workspace_target(target_pos)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        quat_norm = math.sqrt(sum(v * v for v in target_quat))
        mode_used = "full_pose"

        if quat_norm < 1e-6:
            if allow_orientation_fallback:
                self._log.warning("  Quaternion near-zero → position-only fallback")
                with self._runtime.physics_lock:
                    _, cur_quat = ctrl.get_end_effector_pose()
                target_quat = list(cur_quat)
                mode_used = "position_only"
            else:
                return {"status": "error", "message": "Invalid quaternion (near-zero norm)."}
        elif not move_wrist:
            self._log.warning("  move_wrist=False → forcing position-only IK fallback to preserve wrist angle")
            with self._runtime.physics_lock:
                _, cur_quat = ctrl.get_end_effector_pose()
            target_quat = list(cur_quat)
            mode_used = "position_only"
        else:
            target_quat = [v / quat_norm for v in target_quat]

        try:
            with self._runtime.physics_lock:
                if mode_used == "position_only":
                    q_target = ctrl.solve_ik_position_only(target_pos, seed_q_rad, move_wrist=move_wrist)
                else:
                    q_target = ctrl.solve_ik(target_pos, target_quat, seed_q_rad, move_wrist=move_wrist)
        except ValueError as exc:
            self._log.error("  IK / safety error: %s", exc)
            return {"status": "error", "message": f"ik_failed: {exc}"}

        with self._runtime.motion_lock:
            ctrl.send_joint_position_rad(q_target)
            reached = self._runtime.run_until_reached()

        with self._runtime.physics_lock:
            final_pos, final_quat = ctrl.get_end_effector_pose()

        pos_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(final_pos, target_pos)))
        rot_err = None
        if quat_norm >= 1e-6:
            rot_err = _quat_rotation_error(final_quat, tuple(target_quat))

        if pos_err > 0.04:
            status = "error"
            msg = f"ik_failed: position error too large (err={pos_err:.4f}m)"
            self._log.error("  %s", msg)
            return {
                "status": status,
                "message": msg,
                "mode_used": mode_used,
                "pos_err": pos_err,
                "rot_err": rot_err,
            }

        status = "ok" if reached else "timeout"
        self._log.info(
            "  → %s  pos_err=%.4f m  rot_err=%s rad",
            status,
            pos_err,
            f"{rot_err:.4f}" if rot_err is not None else "n/a",
        )
        return {
            "status": status,
            "mode_used": mode_used,
            "q_target_rad": [round(q, 4) for q in q_target],
            "final_pose": {
                "position": [round(p, 4) for p in final_pos],
                "quaternion": [round(q, 4) for q in final_quat],
            },
            "pos_err": round(pos_err, 4),
            "rot_err": round(rot_err, 4) if rot_err is not None else None,
        }

    def rotate_wrist(self, angle_deg: float) -> dict:
        ctrl = self._runtime.get_controller()
        self._log.info("Tool  rotate_wrist(%.1f deg)", angle_deg)

        with self._runtime.motion_lock:
            try:
                ctrl.rotate_wrist(angle_deg)
                reached = self._runtime.run_until_reached()
                status = "ok" if reached else "timeout"
            except Exception as exc:
                self._log.error("rotate_wrist failed: %s", exc)
                return {"status": "error", "message": str(exc)}

        return {"status": status, "message": f"Rotate wrist by {angle_deg} deg finished. Reached={reached}"}

    def get_object_pose(self, body_name: str) -> dict:
        ctrl = self._runtime.get_controller()
        self._log.info("Tool  get_object_pose(%s)", body_name)
        try:
            with self._runtime.physics_lock:
                return ctrl.get_object_pose(body_name)
        except Exception as exc:
            self._log.error("get_object_pose failed: %s", exc)
            return {"status": "error", "message": str(exc)}


class GeometryToolService:
    def compute_grasp_height(
        self,
        geom_type: str,
        size: list[float] | str,
        quat_xyzw: list[float] | str,
    ) -> dict:
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
        else:
            if len(size_list) < 1:
                return {"status": "error", "message": "Sphere size must be [radius]", "top_height": 0.0}
            top_z = float(size_list[0])

        return {"status": "ok", "top_height": round(top_z + 0.04, 4)}

    def compute_wrist_alignment(
        self,
        obj_quat_xyzw: list[float] | str,
        ee_quat_xyzw: list[float] | str,
    ) -> dict:
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


class TaskPlanningToolService:
    def __init__(
        self,
        runtime: ToolRuntimeContext,
        geometry: GeometryToolService,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._runtime = runtime
        self._geometry = geometry
        self._log = logger or logging.getLogger("mcp_kinova")

    def _get_named_pose(self, body_name: str) -> dict:
        ctrl = self._runtime.get_controller()
        with self._runtime.physics_lock:
            return ctrl.get_object_pose(body_name)

    @staticmethod
    def _position_xyz(pose: dict) -> list[float]:
        position = pose["position"]
        return [float(position["x"]), float(position["y"]), float(position["z"])]

    @staticmethod
    def _quat_xyzw(pose: dict) -> list[float]:
        quat = pose["quaternion"]
        return [float(quat["qx"]), float(quat["qy"]), float(quat["qz"]), float(quat["qw"])]

    @staticmethod
    def _position_only_move(
        target_pos: list[float],
        *,
        move_wrist: bool = True,
    ) -> dict:
        return {
            "target_pos": [round(float(v), 4) for v in target_pos],
            "target_quat": [0.0, 0.0, 0.0, 0.0],
            "move_wrist": move_wrist,
        }

    @staticmethod
    def _stack_half_height(pose: dict) -> float:
        geom_type = str(pose.get("geom_type", ""))
        size = [float(v) for v in pose.get("size", [])]
        if geom_type == "box" and len(size) >= 3:
            return size[2]
        if geom_type == "cylinder" and len(size) >= 2:
            return size[1]
        if geom_type in {"sphere", "site"} and size:
            return size[0]
        if size:
            return size[-1]
        return 0.0

    @staticmethod
    def _grasp_profile(profile: str) -> dict[str, float]:
        normalized = str(profile).strip().lower()
        if normalized == "sort_cubes":
            return {"approach_clearance": 0.10, "grasp_offset": 0.015, "lift_clearance": 0.20}
        if normalized == "stack_cubes":
            return {"approach_clearance": 0.15, "grasp_offset": 0.02, "lift_clearance": 0.20}
        if normalized == "generic":
            return {"approach_clearance": 0.15, "grasp_offset": 0.01, "lift_clearance": 0.20}
        return {"approach_clearance": 0.15, "grasp_offset": 0.0, "lift_clearance": 0.20}

    @staticmethod
    def _recommended_gripper_percent(body_name: str, geom_type: str, profile: str) -> float:
        normalized_profile = str(profile).strip().lower()
        normalized_name = body_name.strip().lower()
        if normalized_profile == "sort_cubes":
            return 0.54
        if normalized_profile == "stack_cubes":
            return 0.58
        if geom_type == "sphere":
            return 0.62
        if geom_type == "cylinder":
            return 0.55
        if geom_type == "box":
            return 0.58
        if "cube" in normalized_name:
            return 0.58
        return 0.58

    def plan_object_grasp(self, body_name: str, profile: str = "shapes") -> dict:
        self._log.info("Tool  plan_object_grasp(%s, profile=%s)", body_name, profile)
        pose = self._get_named_pose(body_name)
        if pose.get("status") == "error":
            return pose

        geom_type = str(pose.get("geom_type", "unknown"))
        quat_xyzw = self._quat_xyzw(pose)
        top_height_result = self._geometry.compute_grasp_height(
            geom_type,
            pose.get("size", []),
            quat_xyzw,
        )
        if top_height_result.get("status") != "ok":
            return top_height_result

        profile_params = self._grasp_profile(profile)
        x, y, _ = self._position_xyz(pose)
        top_height = float(top_height_result["top_height"])

        return {
            "status": "ok",
            "body_name": body_name,
            "geom_type": geom_type,
            "top_height": round(top_height, 4),
            "recommended_gripper_percent": round(
                self._recommended_gripper_percent(body_name, geom_type, profile),
                4,
            ),
            "approach_move": self._position_only_move(
                [x, y, top_height + profile_params["approach_clearance"]],
            ),
            "grasp_move": self._position_only_move(
                [x, y, top_height + profile_params["grasp_offset"]],
            ),
            "lift_move": self._position_only_move(
                [x, y, top_height + profile_params["lift_clearance"]],
                move_wrist=False,
            ),
        }

    def plan_wrist_alignment(
        self,
        body_name: str,
        ee_quat_xyzw: list[float] | str,
    ) -> dict:
        self._log.info("Tool  plan_wrist_alignment(%s)", body_name)
        pose = self._get_named_pose(body_name)
        if pose.get("status") == "error":
            return pose

        object_quat = self._quat_xyzw(pose)
        alignment = self._geometry.compute_wrist_alignment(object_quat, ee_quat_xyzw)
        if alignment.get("status") != "ok":
            return alignment

        raw_angle = float(alignment["angle_deg"])
        final_angle = raw_angle
        snapped_for_box = False
        if str(pose.get("geom_type", "")).lower() == "box":
            final_angle = ((raw_angle + 45.0) % 90.0) - 45.0
            snapped_for_box = True

        return {
            "status": "ok",
            "body_name": body_name,
            "geom_type": pose.get("geom_type"),
            "raw_angle_deg": round(raw_angle, 4),
            "angle_deg": round(final_angle, 4),
            "snapped_for_box": snapped_for_box,
        }

    def plan_bin_place(
        self,
        body_name: str,
        target_name: str | None = None,
        profile: str = "sort_cubes",
    ) -> dict:
        del profile
        self._log.info("Tool  plan_bin_place(body=%s, target=%s)", body_name, target_name)
        if target_name is None:
            normalized_name = body_name.strip().lower()
            if "red" in normalized_name:
                target_name = "red_bin_target"
            elif "blue" in normalized_name:
                target_name = "blue_bin_target"
            else:
                return {
                    "status": "error",
                    "message": "Could not infer bin target from body_name. Provide target_name explicitly.",
                }

        target_pose = self._get_named_pose(target_name)
        if target_pose.get("status") == "error":
            return target_pose

        x, y, z = self._position_xyz(target_pose)
        return {
            "status": "ok",
            "body_name": body_name,
            "target_name": target_name,
            "place_move": self._position_only_move([x, y, z]),
            "retreat_move": self._position_only_move([x, y, z + 0.12]),
        }

    def plan_stack_place(
        self,
        bottom_block: str,
        top_block: str,
        profile: str = "stack_cubes",
    ) -> dict:
        del profile
        self._log.info("Tool  plan_stack_place(bottom=%s, top=%s)", bottom_block, top_block)
        bottom_pose = self._get_named_pose(bottom_block)
        if bottom_pose.get("status") == "error":
            return bottom_pose
        top_pose = self._get_named_pose(top_block)
        if top_pose.get("status") == "error":
            return top_pose

        bottom_x, bottom_y, bottom_z = self._position_xyz(bottom_pose)
        bottom_hh = self._stack_half_height(bottom_pose)
        top_hh = self._stack_half_height(top_pose)
        stack_center_z = bottom_z + bottom_hh + top_hh + 0.01

        return {
            "status": "ok",
            "bottom_block": bottom_block,
            "top_block": top_block,
            "stack_center_z": round(stack_center_z, 4),
            "preplace_move": self._position_only_move([bottom_x, bottom_y, stack_center_z + 0.10]),
            "place_move": self._position_only_move([bottom_x, bottom_y, stack_center_z + 0.04]),
            "retreat_move": self._position_only_move(
                [bottom_x, bottom_y, stack_center_z + 0.15],
                move_wrist=False,
            ),
        }

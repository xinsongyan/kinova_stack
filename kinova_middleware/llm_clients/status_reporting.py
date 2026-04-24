from __future__ import annotations

import re

from kinova_middleware.llm_clients.tool_defs import GRAB_SHAPES_TARGETS


async def check_sorting_progress(mcp_client) -> str:
    """Analyze the current scene and report sorting progress to the LLM."""
    bin_centers = {
        "red_bin_target": (0.0, 0.35),
        "blue_bin_target": (0.0, -0.35),
    }
    bin_tolerance = 0.10
    workspace = {
        "red": {"x": (0.18, 0.30), "y": (-0.25, 0.25)},
        "blue": {"x": (-0.30, -0.18), "y": (-0.25, 0.25)},
    }

    try:
        res = await mcp_client.call_tool("get_object_pose", {"body_name": "all"})
        data = res.structured_content or {}
        objects = data.get("objects", [])

        report = []
        actionable = []

        for obj in objects:
            name = obj["body_name"]
            if "cube" not in name.lower():
                continue

            pos = obj["position"]
            x, y = pos["x"], pos["y"]
            color = "red" if "red" in name.lower() else "blue"
            bin_name = f"{color}_bin_target"
            bx, by = bin_centers[bin_name]

            if abs(x - bx) < bin_tolerance and abs(y - by) < bin_tolerance:
                report.append(f"- {name}: Correctly Sorted (In {bin_name})")
                continue

            bounds = workspace[color]
            if bounds["x"][0] <= x <= bounds["x"][1] and bounds["y"][0] <= y <= bounds["y"][1]:
                report.append(f"- {name}: At Starting Position (In {color}_zone - Ready to pick)")
                actionable.append(name)
                continue

            report.append(f"- {name}: OUT OF WORKSPACE / MISPLACED (x={x:.3f}, y={y:.3f})")

        summary = "CURRENT SORTING STATUS REPORT:\n" + ("\n".join(report) if report else "No cubes detected.")
        if not actionable and any("Sorted" not in line for line in report):
            summary += "\n\nWARNING: No actionable cubes found in their workspaces. Remaining cubes are misplaced."
        elif not actionable:
            summary += "\n\nMISSION STATUS: ALL CUBES SORTED SUCCESSFULLY."

        summary += "\n\nCRITICAL SAFETY RULE: Never attempt to pick a cube marked 'OUT OF WORKSPACE'. Focus only on cubes in their designated start zones."
        print(f"\n[check_sorting_status summary]:\n{summary}\n")
        return summary
    except Exception as exc:
        return f"Error checking progress: {str(exc)}"


async def check_stacking_status(mcp_client) -> str:
    """Analyze cube positions and report whether any cube is stacked on another cube."""
    xy_tolerance = 0.05
    min_vertical_gap = 0.045

    try:
        res = await mcp_client.call_tool("get_object_pose", {"body_name": "all"})
        data = res.structured_content or {}
        objects = data.get("objects", [])
        cubes = [obj for obj in objects if "cube" in obj.get("body_name", "").lower()]

        if not cubes:
            return "CURRENT STACKING STATUS REPORT:\nNo cubes detected."

        report = ["CURRENT STACKING STATUS REPORT:"]
        for cube in sorted(cubes, key=lambda item: item["position"]["z"]):
            pos = cube["position"]
            report.append(f"- {cube['body_name']}: x={pos['x']:.3f}, y={pos['y']:.3f}, z={pos['z']:.3f}")

        relations = []
        for top_cube in cubes:
            top_pos = top_cube["position"]
            top_hh = top_cube.get("size", [0.03, 0.03, 0.03])[2] if len(top_cube.get("size", [])) >= 3 else 0.03
            best_candidate = None

            for bottom_cube in cubes:
                if bottom_cube["body_name"] == top_cube["body_name"]:
                    continue

                bottom_pos = bottom_cube["position"]
                bottom_hh = bottom_cube.get("size", [0.03, 0.03, 0.03])[2] if len(bottom_cube.get("size", [])) >= 3 else 0.03
                dx = abs(top_pos["x"] - bottom_pos["x"])
                dy = abs(top_pos["y"] - bottom_pos["y"])
                dz = top_pos["z"] - bottom_pos["z"]
                expected_gap = top_hh + bottom_hh

                if dz <= min_vertical_gap or dx > xy_tolerance or dy > xy_tolerance:
                    continue

                score = abs(dz - expected_gap) + dx + dy
                if best_candidate is None or score < best_candidate["score"]:
                    best_candidate = {
                        "top": top_cube["body_name"],
                        "bottom": bottom_cube["body_name"],
                        "dx": dx,
                        "dy": dy,
                        "dz": dz,
                        "score": score,
                    }

            if best_candidate is not None:
                relations.append(best_candidate)

        unique_relations = {}
        for relation in relations:
            key = (relation["top"], relation["bottom"])
            if key not in unique_relations or relation["score"] < unique_relations[key]["score"]:
                unique_relations[key] = relation

        if unique_relations:
            report.append("")
            report.append("DETECTED STACK RELATIONS:")
            for relation in sorted(unique_relations.values(), key=lambda item: item["dz"], reverse=True):
                report.append(
                    f"- {relation['top']} is on top of {relation['bottom']} "
                    f"(dx={relation['dx']:.3f}, dy={relation['dy']:.3f}, dz={relation['dz']:.3f})"
                )
            if ("blue_cube", "red_cube") in unique_relations:
                report.append("")
                report.append("MISSION STATUS: blue_cube is stacked on red_cube.")
        else:
            report.append("")
            report.append("MISSION STATUS: No cube-on-cube stack detected yet.")

        summary = "\n".join(report)
        print(f"\n[check_stacking_status summary]:\n{summary}\n")
        return summary
    except Exception as exc:
        return f"Error checking stacking status: {str(exc)}"


async def verify_object_lift(mcp_client, body_name: str, min_height: float = 0.12) -> str:
    """Check whether a named object has been lifted high enough above the table."""
    canonical_body_name = str(body_name or "").strip()
    try:
        result = await mcp_client.call_tool("get_object_pose", {"body_name": canonical_body_name})
        data = result.structured_content or {}
        if data.get("status") == "error":
            return f"LIFT CHECK ERROR: {data.get('message', 'Unknown error')}"

        position = data.get("position", {})
        z = float(position.get("z", 0.0))
        lifted = z >= min_height

        summary = (
            f"LIFT STATUS for '{canonical_body_name}': z={z:.3f} m, threshold={min_height:.3f} m. "
            f"{'PASS: object is lifted high enough.' if lifted else 'FAIL: object is still too low.'}"
        )
        print(f"\n[verify_object_lift summary]:\n{summary}\n")
        return summary
    except Exception as exc:
        return f"LIFT CHECK ERROR for '{canonical_body_name}': {str(exc)}"


def record_grab_verification(verification_history: dict, body_name: str, result: str, min_height: float) -> None:
    """Store the official outcome of a verify_object_lift tool call."""
    canonical_body_name = str(body_name or "").strip()
    status = "error"
    passed = False
    z_height = None

    if "PASS:" in result:
        status = "pass"
        passed = True
    elif "FAIL:" in result:
        status = "fail"

    match = re.search(r"z=([0-9.+-]+)", result)
    if match:
        try:
            z_height = float(match.group(1))
        except ValueError:
            z_height = None

    event = {
        "status": status,
        "passed": passed,
        "z_height": z_height,
        "min_height": float(min_height),
        "raw_result": result,
    }
    history = verification_history.setdefault(canonical_body_name, {"attempts": []})
    history["attempts"].append(event)


async def collect_grab_shapes_report(
    mcp_client,
    verification_history: dict,
    targets: tuple[str, ...] = GRAB_SHAPES_TARGETS,
    default_min_height: float = 0.12,
) -> dict:
    """Build a scene-grounded final report for the grab_shapes workflow."""
    objects = []
    verified_count = 0

    normalized_targets = tuple(str(body_name or "").strip() for body_name in targets)

    for body_name in normalized_targets:
        history = verification_history.get(body_name, {})
        attempts = history.get("attempts", [])
        successful_attempt = next((attempt for attempt in reversed(attempts) if attempt.get("passed")), None)
        latest_attempt = attempts[-1] if attempts else None

        try:
            result = await mcp_client.call_tool("get_object_pose", {"body_name": body_name})
            data = result.structured_content or {}
        except Exception as exc:
            data = {"status": "error", "message": str(exc)}

        if data.get("status") == "error":
            objects.append(
                {
                    "body_name": body_name,
                    "officially_picked_up": bool(successful_attempt),
                    "attempt_count": len(attempts),
                    "latest_attempt": latest_attempt,
                    "successful_attempt": successful_attempt,
                    "current_pose_error": data.get("message", "Unknown error"),
                }
            )
            if successful_attempt:
                verified_count += 1
            continue

        position = data.get("position", {})
        current_z = float(position.get("z", 0.0))
        min_height = (
            successful_attempt.get("min_height")
            if successful_attempt is not None
            else latest_attempt.get("min_height", default_min_height)
            if latest_attempt is not None
            else float(default_min_height)
        )
        currently_elevated = current_z >= min_height

        object_report = {
            "body_name": body_name,
            "officially_picked_up": bool(successful_attempt),
            "attempt_count": len(attempts),
            "latest_attempt": latest_attempt,
            "successful_attempt": successful_attempt,
            "current_pose": {
                "x": float(position.get("x", 0.0)),
                "y": float(position.get("y", 0.0)),
                "z": current_z,
            },
            "currently_elevated": currently_elevated,
            "min_height": float(min_height),
        }
        objects.append(object_report)
        if successful_attempt:
            verified_count += 1

    return {
        "workflow": "grab_shapes",
        "targets": list(normalized_targets),
        "verified_count": verified_count,
        "target_count": len(normalized_targets),
        "objects": objects,
    }


def format_grab_shapes_report(report: dict) -> str:
    """Format a grab_shapes verification report for terminal output."""
    lines = ["OFFICIAL GRAB REPORT:"]

    for obj in report.get("objects", []):
        name = obj["body_name"]
        attempt_count = obj.get("attempt_count", 0)
        successful_attempt = obj.get("successful_attempt")
        latest_attempt = obj.get("latest_attempt")

        if obj.get("current_pose_error"):
            status = "OFFICIALLY PICKED UP" if obj.get("officially_picked_up") else "NOT OFFICIALLY PICKED UP"
            lines.append(
                f"- {name}: {status}. Could not read final pose: {obj['current_pose_error']} "
                f"(verify attempts={attempt_count})."
            )
            continue

        pose = obj["current_pose"]
        if successful_attempt:
            if obj.get("currently_elevated"):
                final_state = "still elevated at the end of the run"
            else:
                final_state = "not elevated at the end of the run"
            lines.append(
                f"- {name}: OFFICIALLY PICKED UP. verify_object_lift passed "
                f"(z={successful_attempt.get('z_height', 0.0):.3f} m, threshold={obj['min_height']:.3f} m). "
                f"Final pose=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}) and the object is {final_state}. "
                f"(verify attempts={attempt_count})"
            )
            continue

        if latest_attempt and latest_attempt.get("status") == "fail":
            verification_note = (
                f"latest verify_object_lift failed "
                f"(z={latest_attempt.get('z_height', 0.0):.3f} m, threshold={latest_attempt.get('min_height', obj['min_height']):.3f} m)"
            )
        elif latest_attempt and latest_attempt.get("status") == "error":
            verification_note = "latest verify_object_lift returned an error"
        else:
            verification_note = "verify_object_lift was never called successfully"

        lines.append(
            f"- {name}: NOT OFFICIALLY PICKED UP. {verification_note}. "
            f"Final pose=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}). "
            f"(verify attempts={attempt_count})"
        )

    lines.append("")
    lines.append(
        f"MISSION STATUS: Officially picked up {report.get('verified_count', 0)}/{report.get('target_count', 0)} target objects."
    )
    return "\n".join(lines)

#!/usr/bin/env python3
"""
Stack All Cubes — Autonomous pick-and-place stacking demo.

Connects to the Kinova MCP server, discovers all cubes in the scene,
and stacks them into a neat vertical tower at a chosen destination.

Uses only MCP server tools — no internal helper functions.

Usage:
    1. Start server:  mjpython kinova_middleware/backend/mcp_kinova_server.py
    2. Run stacker:   python kinova_middleware/stack_all_cubes.py
"""
from __future__ import annotations

import asyncio
import math
from fastmcp import Client

# ── Configuration ────────────────────────────────────────────────────────────
SERVER_URL = "http://127.0.0.1:8000/mcp"

DEST_X, DEST_Y = 0.20, -0.20       # tower destination (world XY)

PREGRASP_Z   = 0.08                 # above cube top before descending
GRASP_OFFSET = 0.008                # above cube top when grasping
LIFT_Z       = 0.20                 # above cube top after grasping
PREPLACE_Z   = 0.10                 # above target z before descending
PLACE_Z      = 0.04                 # above target z when releasing
RETREAT_Z    = 0.10                 # above target z after releasing

STACK_MARGIN = 0.002                # gap between stacked layers
MAX_STACK_Z  = 0.30                 # workspace ceiling

POS_QUAT = [0.0, 0.0, 0.0, 0.0]    # triggers position-only IK
GRIP_OPEN = 0.9
GRIP_CLOSE = 0.58


# ── Helpers ──────────────────────────────────────────────────────────────────

def p(result) -> dict:
    return result.structured_content or {}


def log(msg: str, indent: int = 0):
    print(f"{'  ' * indent}{msg}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    async with Client(SERVER_URL) as client:
        log("=" * 55)
        log("  STACK ALL CUBES")
        log("=" * 55)

        # ── Home ─────────────────────────────────────────────
        log("\n[0] Homing …")
        await client.call_tool("move_home")
        await client.call_tool("set_gripper", {"percent": GRIP_OPEN})
        log("✓ Home + gripper open.\n")

        # ── Discover cubes ───────────────────────────────────
        log("[1] Discovering cubes …")
        r = p(await client.call_tool("get_object_pose", {"body_name": "all"}))
        cubes = [o for o in r.get("objects", [])
                 if o.get("geom_type") == "box" or "cube" in o.get("body_name", "").lower()]

        if not cubes:
            log("✗ No cubes found.")
            return

        for c in cubes:
            pos = c["position"]
            log(f"  • {c['body_name']:15s}  ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")

        # Estimate table height
        table_z = min(c["position"]["z"] - (c["size"][2] if len(c.get("size", [])) >= 3 else 0.03)
                      for c in cubes)

        # Sort: closest to destination first
        cubes.sort(key=lambda c: math.hypot(c["position"]["x"] - DEST_X,
                                            c["position"]["y"] - DEST_Y))

        # ── Pick-and-place loop ──────────────────────────────
        stack = 0
        results = []

        for i, cube in enumerate(cubes):
            name = cube["body_name"]
            size = cube.get("size", [0.03, 0.03, 0.03])
            hh = size[2] if len(size) >= 3 else 0.03   # half-height
            ch = 2.0 * hh                               # full height

            target_z = table_z + hh + stack * (ch + STACK_MARGIN)
            if target_z > MAX_STACK_Z:
                log(f"⚠ Stack height {target_z:.3f}m exceeds max — stopping.")
                break

            log(f"\n{'─'*50}")
            log(f"  [{i+1}/{len(cubes)}] {name}  → layer {stack}")
            log(f"{'─'*50}")

            # ── Re-query position ────────────────────────────
            fresh = p(await client.call_tool("get_object_pose", {"body_name": name}))
            if fresh.get("status") == "error":
                log(f"  ⚠ {name} not found — skip", 1)
                results.append((name, "missing"))
                continue
            pos = fresh["position"]
            x, y, z = pos["x"], pos["y"], pos["z"]
            z_top = z + hh
            obj_q = fresh.get("quaternion", {})
            obj_quat = [obj_q.get("qx", 0), obj_q.get("qy", 0),
                        obj_q.get("qz", 0), obj_q.get("qw", 1)]

            # ═══════════════ PICK ════════════════════════════
            # 1. Open gripper
            await client.call_tool("set_gripper", {"percent": GRIP_OPEN})

            # 2. Pregrasp (position-only)
            r = p(await client.call_tool("move_pose", {
                "target_pos": [x, y, z_top + PREGRASP_Z],
                "target_quat": POS_QUAT,
            }))
            log(f"  Pregrasp  err={r.get('pos_err', 0):.4f}m", 1)

            # 3. Align wrist with cube
            ee = p(await client.call_tool("get_end_effector_pose"))
            ee_q = ee.get("quaternion", {})
            ee_quat = [ee_q.get("qx", 0), ee_q.get("qy", 0),
                       ee_q.get("qz", 0), ee_q.get("qw", 1)]
            align = p(await client.call_tool("compute_wrist_alignment", {
                "obj_quat_xyzw": obj_quat, "ee_quat_xyzw": ee_quat,
            }))
            angle = align.get("angle_deg", 0)
            if abs(angle) > 1.0:
                log(f"  Wrist align {angle:.1f}°", 1)
                await client.call_tool("rotate_wrist", {"angle_deg": angle})

            # 4. Descend
            r = p(await client.call_tool("move_pose", {
                "target_pos": [x, y, z_top + GRASP_OFFSET],
                "target_quat": POS_QUAT,
            }))
            log(f"  Descend   err={r.get('pos_err', 0):.4f}m", 1)

            # 5. Grasp
            log(f"  Grasping …", 1)
            await client.call_tool("set_gripper", {"percent": GRIP_CLOSE})
            await asyncio.sleep(5.0)

            # 6. Record grasp orientation
            ee2 = p(await client.call_tool("get_end_effector_pose"))
            gq = ee2.get("quaternion", {})
            grasp_quat = [gq.get("qx", 0), gq.get("qy", 0),
                          gq.get("qz", 0), gq.get("qw", 1)]

            # 7. Lift (maintain orientation)
            r = p(await client.call_tool("move_pose", {
                "target_pos": [x, y, z_top + LIFT_Z],
                "target_quat": grasp_quat,
            }))
            log(f"  Lift      err={r.get('pos_err', 0):.4f}m", 1)

            # 8. Verify grasp
            await asyncio.sleep(0.3)
            check = p(await client.call_tool("get_object_pose", {"body_name": name}))
            new_z = check.get("position", {}).get("z", 0)
            if new_z < z + 0.02:
                log(f"  ✗ Grasp failed (z {z:.3f} → {new_z:.3f})", 1)
                await client.call_tool("set_gripper", {"percent": GRIP_OPEN})
                results.append((name, "pick_failed"))
                await client.call_tool("move_home")
                continue
            log(f"  ✓ Grasp OK (z {z:.3f} → {new_z:.3f})", 1)

            # ═══════════════ PLACE ═══════════════════════════
            # 9. Preplace (maintain orientation)
            r = p(await client.call_tool("move_pose", {
                "target_pos": [DEST_X, DEST_Y, target_z + PREPLACE_Z],
                "target_quat": grasp_quat,
            }))
            log(f"  Preplace  err={r.get('pos_err', 0):.4f}m", 1)

            # 10. Place descend
            r = p(await client.call_tool("move_pose", {
                "target_pos": [DEST_X, DEST_Y, target_z + PLACE_Z],
                "target_quat": grasp_quat,
            }))
            log(f"  Place     err={r.get('pos_err', 0):.4f}m", 1)

            # 11. Release
            await client.call_tool("set_gripper", {"percent": GRIP_OPEN})
            await asyncio.sleep(0.3)

            # 12. Retreat
            r = p(await client.call_tool("move_pose", {
                "target_pos": [DEST_X, DEST_Y, target_z + RETREAT_Z],
                "target_quat": grasp_quat,
            }))
            log(f"  Retreat   err={r.get('pos_err', 0):.4f}m", 1)

            log(f"  ✓ Placed at layer {stack}", 1)
            stack += 1
            results.append((name, "stacked"))

            # Home between cubes
            await client.call_tool("move_home")

        # ── Summary ──────────────────────────────────────────
        await client.call_tool("move_home")
        log(f"\n{'='*55}")
        log(f"  DONE — {stack} cube(s) stacked at ({DEST_X}, {DEST_Y})")
        log(f"{'='*55}\n")
        for name, status in results:
            icon = "✓" if status == "stacked" else "✗"
            log(f"  {icon} {name:15s} → {status}")


if __name__ == "__main__":
    asyncio.run(main())

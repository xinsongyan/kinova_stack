#!/usr/bin/env python3
"""
Sort Cubes — Autonomous sorting demo.

Connects to the Kinova MCP server, discovers all cubes in the scene,
and sorts them into the corresponding bins (red into red bin, blue into blue bin).

Uses only MCP server tools — no internal helper functions.

Usage:
    1. Start server:  KINOVA_SCENE=sorting_task.xml mjpython kinova_middleware/backend/mcp_server/app.py
    2. Run sorter:    python kinova_middleware/sort_cubes.py
"""
from __future__ import annotations

import asyncio
import math
from fastmcp import Client

# ── Configuration ────────────────────────────────────────────────────────────
SERVER_URL = "http://127.0.0.1:8000/mcp"

# Clearances (metres)
PREGRASP_Z   = 0.08
GRASP_OFFSET = 0.022
LIFT_Z       = 0.20

POS_QUAT = [0.0, 0.0, 0.0, 0.0]

GRIP_OPEN  = 0.9
GRIP_CLOSE = 0.55


# ── Helpers ──────────────────────────────────────────────────────────────────

def p(result) -> dict:
    return result.structured_content or {}


def log(msg: str, indent: int = 0):
    print(f"{'  ' * indent}{msg}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    async with Client(SERVER_URL) as client:
        log("=" * 55)
        log("  SORT CUBES — Autonomous Sorting")
        log("=" * 55)

        # ── Home ─────────────────────────────────────────────
        log("\n[0] Homing …")
        await client.call_tool("move_home")
        await client.call_tool("set_gripper", {"percent": GRIP_OPEN})
        log("✓ Home + gripper open.\n")

        # ── Discover cubes ───────────────────────────────────
        log("[1] Discovering cubes …")
        r = p(await client.call_tool("get_object_pose", {"body_name": "all"}))
        cubes = [o for o in r.get("objects", []) if "cube" in o.get("body_name", "").lower()]

        if not cubes:
            log("✗ No cubes found.")
            return

        red_cubes = sorted(
            [c for c in cubes if "red" in c["body_name"].lower()],
            key=lambda c: math.hypot(c["position"]["x"], c["position"]["y"]),
        )
        blue_cubes = sorted(
            [c for c in cubes if "blue" in c["body_name"].lower()],
            key=lambda c: math.hypot(c["position"]["x"], c["position"]["y"]),
        )
        queue = blue_cubes + red_cubes

        log(f"Found {len(red_cubes)} red, {len(blue_cubes)} blue cube(s).\n")
        for c in queue:
            pos = c["position"]
            log(f"  • {c['body_name']:15s}  ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")

        # ── Sort loop ────────────────────────────────────────
        results = []

        for i, cube in enumerate(queue):
            name = cube["body_name"]
            is_red = "red" in name.lower()
            dest_target = "red_bin_target" if is_red else "blue_bin_target"
            size = cube.get("size", [0.03, 0.03, 0.03])
            hh = size[2] if len(size) >= 3 else 0.03

            log(f"\n{'─'*50}")
            log(f"  [{i+1}/{len(queue)}] {name}  → {dest_target}")
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
                "target_quat": POS_QUAT,
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

            # ═══════════════ DROP INTO BIN ═══════════════════
            # 9. Get bin target position
            bin_data = p(await client.call_tool("get_object_pose", {"body_name": dest_target}))
            if bin_data.get("status") == "error":
                log(f"  ✗ Bin target {dest_target} not found", 1)
                await client.call_tool("set_gripper", {"percent": GRIP_OPEN})
                results.append((name, "place_failed"))
                await client.call_tool("move_home")
                continue
            bp = bin_data["position"]
            bx, by, bz = bp["x"], bp["y"], bp["z"]

            # 10. Move over bin (position-only)
            r = p(await client.call_tool("move_pose", {
                "target_pos": [bx, by, bz],
                "target_quat": POS_QUAT,
            }))
            log(f"  Over bin  err={r.get('pos_err', 0):.4f}m", 1)

            # 11. Release
            log(f"  Dropping …", 1)
            await client.call_tool("set_gripper", {"percent": GRIP_OPEN})
            await asyncio.sleep(0.5)

            log(f"  ✓ Sorted into {dest_target}", 1)
            results.append((name, "sorted"))

            # Home between cubes
            await client.call_tool("move_home")

        # ── Verification ─────────────────────────────────────
        log("\n[2] Verifying sorted cubes …")
        await asyncio.sleep(1.0) # Let physics settle
        
        BIN_CENTERS = {
            "red_bin_target": (0.0, 0.35),
            "blue_bin_target": (0.0, -0.35)
        }
        BIN_TOLERANCE = 0.08  # Walls are ~0.09m from center, cube radius is 0.0125m

        verified_results = []
        for name, status in results:
            if status == "sorted":
                final = p(await client.call_tool("get_object_pose", {"body_name": name}))
                if final.get("status") == "error":
                    verified_results.append((name, "missing_at_verification"))
                    continue
                
                pos = final["position"]
                x, y = pos["x"], pos["y"]
                dest = "red_bin_target" if "red" in name.lower() else "blue_bin_target"
                bx, by = BIN_CENTERS[dest]
                
                # Check if center of cube is strictly within the walls' XY bounds
                if abs(x - bx) <= BIN_TOLERANCE and abs(y - by) <= BIN_TOLERANCE:
                     log(f"  ✓ {name:15s} verified inside {dest}", 1)
                     verified_results.append((name, "sorted_verified"))
                else:
                     log(f"  ✗ {name:15s} fell out! (x={x:.3f}, y={y:.3f})", 1)
                     verified_results.append((name, "missed_bin"))
            else:
                verified_results.append((name, status))
        
        results = verified_results

        # ── Summary ──────────────────────────────────────────
        await client.call_tool("move_home")
        n_sorted = sum(1 for _, s in results if s == "sorted_verified")
        log(f"\n{'='*55}")
        log(f"  DONE — {n_sorted}/{len(queue)} cube(s) verified in bins")
        log(f"{'='*55}\n")
        for name, status in results:
            icon = "✓" if status == "sorted_verified" else "✗"
            log(f"  {icon} {name:15s} → {status}")


if __name__ == "__main__":
    asyncio.run(main())

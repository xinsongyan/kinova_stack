#!/usr/bin/env python3
"""IK verification via the MCP server.

Connects to the running MCP server, discovers all target_* sites,
moves the arm to each one, and reports position errors.

Usage:  python ik_verify_mcp.py
(Requires the MCP server to be running first.)
"""

import asyncio
import csv
import math
import os
import random
from dataclasses import dataclass, field
from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"

# Downward-pointing quaternion (Rotates World X to World -Z)
DOWN_QUAT = [0.0, 0.70710678, 0.0, 0.70710678]  # qx, qy, qz, qw

POS_ERROR_THRESHOLD = 0.01  # 10 mm
CSV_PATH = os.path.join(os.path.dirname(__file__), "ik_mcp_results.csv")

TARGET_NAMES = [f"target_{i:02d}" for i in range(1, 21)]
random.shuffle(TARGET_NAMES)
print(TARGET_NAMES)


def p(result) -> dict:
    return result.structured_content or {}


@dataclass
class Result:
    name: str = ""
    target_pos: list = field(default_factory=list)
    reached_pos: list = field(default_factory=list)
    pos_error: float = 0.0
    mode_used: str = ""
    status: str = ""
    success: bool = False


async def main():
    print("=" * 55)
    print("  IK Verification via MCP Server")
    print("=" * 55)

    async with Client(SERVER_URL) as client:
        print("  Connected to MCP server.\n")

        # Move home first
        print("  Moving HOME...", end=" ", flush=True)
        await client.call_tool("move_home")
        print("done.")



        results: list[Result] = []

        for i, tgt_name in enumerate(TARGET_NAMES):
            r = Result(name=tgt_name)

            # Get target position from scene site
            pose_data = p(await client.call_tool("get_object_pose", {"body_name": tgt_name}))
            if pose_data.get("status") == "error":
                print(f"  [{i+1}/{len(TARGET_NAMES)}] {tgt_name}: SKIP (not found)")
                r.status = "not_found"
                results.append(r)
                continue

            pos = pose_data["position"]
            tx, ty, tz = pos["x"], pos["y"], pos["z"]
            r.target_pos = [tx, ty, tz]

            print(f"  [{i+1}/{len(TARGET_NAMES)}] {tgt_name} → "
                  f"({tx:+.3f}, {ty:+.3f}, {tz:+.3f})", end="  ", flush=True)

            # Move to target using IK
            move_data = p(await client.call_tool("move_pose", {
                "target_pos": [tx, ty, tz],
                "target_quat": DOWN_QUAT,
                "allow_orientation_fallback": True,
            }))

            r.status = move_data.get("status", "error")
            r.mode_used = move_data.get("mode_used", "?")
            r.pos_error = move_data.get("pos_err", 999.0)

            # Read final EE pose
            ee = p(await client.call_tool("get_end_effector_pose"))
            ep = ee.get("position", {})
            r.reached_pos = [ep.get("x", 0), ep.get("y", 0), ep.get("z", 0)]

            r.success = r.status == "ok" and r.pos_error <= POS_ERROR_THRESHOLD

            tag = "✓" if r.success else "✗"
            print(f"{tag}  err={r.pos_error:.4f}m  [{r.mode_used}]  {r.status}")



            results.append(r)

        # ── Summary ──────────────────────────────────────────────
        n_ok = sum(1 for r in results if r.success)
        errs = [r.pos_error for r in results if r.status != "not_found"]

        print(f"\n{'=' * 55}")
        print(f"  RESULTS: {n_ok}/{len(results)} passed")
        print(f"  Avg pos error: {sum(errs)/len(errs):.4f} m" if errs else "")
        print(f"  Max pos error: {max(errs):.4f} m" if errs else "")

        failed = [r for r in results if not r.success]
        if failed:
            print(f"  Failed ({len(failed)}):")
            for r in failed:
                print(f"    {r.name}: err={r.pos_error:.4f}m  status={r.status}")

        print(f"{'=' * 55}")

        # ── CSV ──────────────────────────────────────────────────
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "tx", "ty", "tz", "rx", "ry", "rz",
                         "pos_err", "mode", "status", "pass"])
            for r in results:
                tp = r.target_pos or [0, 0, 0]
                rp = r.reached_pos or [0, 0, 0]
                w.writerow([r.name, *tp, *rp, f"{r.pos_error:.6f}",
                            r.mode_used, r.status, r.success])
        print(f"  CSV → {CSV_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

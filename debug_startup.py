
import sys
import os
print("DEBUG: Starting...", file=sys.stderr, flush=True)

try:
    print("DEBUG: Importing mujoco...", file=sys.stderr, flush=True)
    import mujoco
    print("DEBUG: Mujoco imported.", file=sys.stderr, flush=True)

    print("DEBUG: Importing fastmcp...", file=sys.stderr, flush=True)
    import fastmcp
    print("DEBUG: FastMCP imported.", file=sys.stderr, flush=True)

    # Setup path like mcp_kinova_server does
    _THIS_DIR = os.path.abspath("kinova_middleware/backend")
    _ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
    sys.path.append(_ROOT_DIR)
    
    print(f"DEBUG: Added {_ROOT_DIR} to path.", file=sys.stderr, flush=True)
    sys.path.append(_THIS_DIR)
    print(f"DEBUG: Added {_THIS_DIR} to path.", file=sys.stderr, flush=True)

    print("DEBUG: Importing kinova_controller...", file=sys.stderr, flush=True)
    import kinova_middleware.backend.kinova_controller
    print("DEBUG: kinova_controller imported.", file=sys.stderr, flush=True)

    print("DEBUG: Importing kinova_mujoco_backend...", file=sys.stderr, flush=True)
    import kinova_middleware.backend.kinova_mujoco_backend
    print("DEBUG: kinova_mujoco_backend imported.", file=sys.stderr, flush=True)

    print("DEBUG: Importing kinova_sdk_backend...", file=sys.stderr, flush=True)
    import kinova_middleware.backend.kinova_sdk_backend
    print("DEBUG: kinova_sdk_backend imported.", file=sys.stderr, flush=True)

    print("DEBUG: Importing mcp_kinova_server...", file=sys.stderr, flush=True)
    import kinova_middleware.backend.mcp_kinova_server
    print("DEBUG: mcp_kinova_server imported.", file=sys.stderr, flush=True)

except Exception as e:
    print(f"DEBUG: Exception: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()

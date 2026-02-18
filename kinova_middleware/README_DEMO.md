# Kinova Semi-Circle Demo

This project demonstrates LLM-driven robot control using [FastMCP](https://github.com/punkpeye/fastmcp) and Google Gemini.

## Components

1. **`mcp_server_demo.py`**: A specialized MCP server that wraps the Kinova MuJoCo simulation.
   - Exposes blocking tools (commands wait for physical completion).
   - Supports position-only IK fallback.
   - Runs a 200Hz physics loop in the background.

2. **`gemini_semicircle_client.py`**: A Python client using `google-genai`.
   - Takes 3 calibration points (P0, P90, P180).
   - Infers a semi-circle path.
   - Commands the robot to trace the path point-by-point.

## Usage

### 1. Start the Server
Run this in a terminal. It will open a MuJoCo viewer window.
```bash
# From kinova_stack root
source .venv/bin/activate
python kinova_middleware/mcp_server_demo.py --transport streamable-http --viewer
```
Wait until you see `[Demo Server] Ready...`

### 2. Run the Client
In a separate terminal:
```bash
# From kinova_stack root
source .venv/bin/activate
export GOOGLE_API_KEY="your-api-key"
python kinova_middleware/gemini_semicircle_client.py
```

## How it works
The client sends the calibration points to Gemini. Gemini calculates intermediate waypoints for a perfect semi-circle and calls `move_pose` for each one. The robot moves, and the server only replies when the motion is stable. If IK fails for a specific orientation, the client/server logic allows falling back to position-only control or retrying.

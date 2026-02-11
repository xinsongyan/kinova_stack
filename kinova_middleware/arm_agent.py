"""
Gemini + FastMCP agent that turns natural language into MCP tool calls to move the Kinova Gen3 arm.

The agent:
1. Connects to the running MCP server over HTTP (streamable transport).
2. Sends a natural language request to Gemini with the MCP tools attached.
3. Gemini chooses the right joint tool and moves it (e.g., to 90 degrees = 1.5708 rad).

Prereqs:
- fastmcp
- google-genai (Gemini SDK)
- mujoco + numpy (running MCP server: `python mcp_server/robot_arm_server.py --transport streamable-http --viewer`)
"""
import asyncio
import argparse
import os

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from google import genai


SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
ALLOWED_FUNCS = [
    "open_hand",
    "close_hand",
    "set_cartesian_pose",
    "enqueue_pose",
    "queue_status",
    "clear_queue",
    "get_state",
]

# Configure Gemini client (expects GOOGLE_API_KEY env var).
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Set GOOGLE_API_KEY in your environment to use Gemini.")

# Connect to the already-running MCP server over HTTP (streamable).
mcp_client = Client(StreamableHttpTransport(url=SERVER_URL))
gemini_client = genai.Client(api_key=api_key)


SYSTEM_PROMPT = (
    "You control a simplified Kinova Gen3 arm simulated in MuJoCo. The robot exposes 10 revolute joints: "
    "joint_1..joint_4 drive the arm, and joint_finger_{1,2,3} plus joint_finger_tip_{1,2,3} drive the hand.\n\n"
    "=== CONTROLLER WORKFLOW ===\n"
    "- The MCP tools do not move the robot instantly; they enqueue FIFO tasks that the controller executes one at a time. "
    "The controller idles for two seconds when the queue is empty.\n"
    "- Use `open_hand(intensity, duration)` to open the hand (intensity=1.0 is fully open) and "
    "`close_hand(intensity, duration)` to close it (intensity=1.0 is fully closed). Intensities are floats clipped to [0,1].\n"
    "- Use `set_cartesian_pose(position, orientation, duration)` to point the arm toward a Cartesian target. "
    "`position` is an [x,y,z] point in meters relative to the base, and `orientation` is an optional [roll,pitch,yaw] "
    "that the controller approximates by tweaking joint_4.\n"
    "- Use `enqueue_pose(values, duration)` to move all 10 joints directly when you need a particular configuration. "
    "Each duration controls how long the controller takes to blend toward the target.\n"
    "- Call `queue_status()` or `get_state()` to monitor progress, and use `clear_queue(drop_active=True)` if you need to cancel "
    "pending or running tasks.\n\n"
    "Always respond with a single MCP tool call that matches the user's intent."
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Natural language to MCP robot control.")
    parser.add_argument(
        "text",
        nargs="*",
        help="Natural language command (default: move shoulder lift to 90 degrees).",
    )
    args = parser.parse_args()
    user_request = "Move the arm to an up pose."

    def _response_has_tool_call(resp: genai.types.GenerateContentResponse) -> bool:
        if resp.automatic_function_calling_history:
            return True
        for candidate in resp.candidates or []:
            for part in candidate.content.parts or []:
                if part.function_call:
                    return True
        return False

    async with mcp_client:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"{SYSTEM_PROMPT}\nUser request: {user_request}",
            config=genai.types.GenerateContentConfig(
                temperature=0,
                tools=[mcp_client.session],
                tool_config=genai.types.ToolConfig(
                    function_calling_config=genai.types.FunctionCallingConfig(
                        mode=genai.types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=ALLOWED_FUNCS,
                    )
                ),
            ),
        )
        if not _response_has_tool_call(response):
            raise RuntimeError("Model response did not include an MCP tool call. Check the prompt/tool_config.")
        print(response.text)


if __name__ == "__main__":
    asyncio.run(main())

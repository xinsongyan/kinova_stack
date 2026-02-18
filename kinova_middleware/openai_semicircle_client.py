#!/usr/bin/env python3
"""
OpenAI Client for Kinova Semi-Circle Demo.

Connects to the demo MCP server and instructs GPT-4o to move the robot
along a semi-circle path defined by 3 calibration points.

Equivalent to gemini_semicircle_client.py but using the OpenAI API.

Usage:
    1. Start MCP server:  .venv/bin/mjpython kinova_middleware/mcp_kinova_server.py
    2. Run this client:   python kinova_middleware/openai_semicircle_client.py
"""
import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from openai import AsyncOpenAI, RateLimitError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SERVER_URL = "http://127.0.0.1:8000/mcp"

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
API_KEY = os.getenv("OPEN_AI_API_KEY")

if not API_KEY:
    print("Error: OPEN_AI_API_KEY not set in .env or environment.", file=sys.stderr)
    sys.exit(1)

MODEL = "gpt-4o"  # or "gpt-4o-mini" for faster/cheaper runs

# ---------------------------------------------------------------------------
# Calibration Data (P0, P90, P180) — same as Gemini client
# ---------------------------------------------------------------------------
CALIBRATION_POINTS = """
- P0  (0 deg):   pos = [0.337, -0.006, 0.166]
- P90 (90 deg):  pos = [0.159, -0.244, 0.167]
- P180(180 deg): pos = [-0.222, -0.146, 0.167]
"""

SYSTEM_PROMPT = f"""
You are an expert robot controller. You control a Kinova arm via MCP tools.
Your task is to move the robot end-effector along a SEMI-CIRCLE path.

### WORKSPACE CALIBRATION
You are given exactly three points on the arc:
{CALIBRATION_POINTS}

### INSTRUCTIONS
1. **Analyze Geometry**: Fit a semi-circle passing through P0 -> P90 -> P180.
   - The path must be a circular arc in 3D space.
   - Calculate the center and radius.
   - Interpolate waypoints along this arc.

2. **Generate Path**:
   - Create a sequence of waypoints at 10-degree increments (0, 10, 20, ..., 180).
   - Ensure Z height remains roughly constant (~0.167).

3. **Execute Motion**:
   - Use `move_pose(pos=[x,y,z], quat=...)` for each waypoint sequentially.
   - **BLOCKING**: You must wait for each tool call to return "reached" before sending the next.
   - **ORIENTATION**:
     - First, call `get_state()` to see the current orientation.
     - You may try to maintain that orientation for the whole path.
     - IF `move_pose` fails (IK error), retry that waypoint with `quat=None` (position-only fallback).
   
4. **Safety**:
   - If a point is unreachable, try moving it slightly (1-2cm) towards the circle center and retry.
   - If completely stuck, stop and report error.

### CONSTRAINTS
- Output ONLY tool calls. Do not write explanations until the very end.
- Call tools sequentially (one by one).
"""


def mcp_tools_to_openai(tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function-calling format."""
    openai_tools = []
    for t in tools:
        # Build JSON schema from MCP tool input schema
        params = t.inputSchema if t.inputSchema else {"type": "object", "properties": {}}
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": params,
            },
        })
    return openai_tools


async def chat_completion_with_retry(client, messages, tools, max_retries=5, initial_delay=5.0):
    """Call OpenAI chat completion with retry for rate limits (429)."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
            )
        except RateLimitError:
            print(f"Rate limit hit (429). Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
            await asyncio.sleep(delay)
            delay *= 1.5
    raise RuntimeError(f"Max retries ({max_retries}) exceeded for OpenAI API call.")


async def main():
    print(f"Connecting to MCP Server at {SERVER_URL}...")

    # MCP client
    try:
        mcp_client = Client(StreamableHttpTransport(url=SERVER_URL))
    except Exception as e:
        print(f"Failed to connect to MCP server: {e}")
        return

    # OpenAI client
    openai_client = AsyncOpenAI(api_key=API_KEY)

    async with mcp_client:
        # Fetch MCP tools and convert to OpenAI format
        mcp_tools = await mcp_client.list_tools()
        openai_tools = mcp_tools_to_openai(mcp_tools)

        print(f"Loaded {len(openai_tools)} tools: {[t['function']['name'] for t in openai_tools]}")
        print(f"Using model: {MODEL}")
        print()

        # Build initial messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Begin the semi-circle demonstration."},
        ]
        print("User: Begin the semi-circle demonstration.")

        # Conversation loop
        while True:
            response = await chat_completion_with_retry(openai_client, messages, openai_tools)
            choice = response.choices[0]

            # If the model is done (no tool calls)
            if choice.finish_reason == "stop":
                print(f"\nModel: {choice.message.content}")
                break

            # The model wants to call tools
            assistant_msg = choice.message
            messages.append(assistant_msg)  # append the assistant message with tool_calls

            if not assistant_msg.tool_calls:
                # Model finished with content
                print(f"\nModel: {assistant_msg.content}")
                break

            # Execute each tool call
            for tool_call in assistant_msg.tool_calls:
                fname = tool_call.function.name
                fargs_str = tool_call.function.arguments
                fargs = json.loads(fargs_str) if fargs_str else {}

                print(f" >> Tool Call: {fname}({fargs})")

                # Execute via MCP
                try:
                    result = await mcp_client.call_tool(fname, fargs)
                    output_text = str(result)
                except Exception as e:
                    output_text = f"Error: {str(e)}"

                print(f"    Result: {output_text[:200]}...")

                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output_text,
                })

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Gemini Client for Kinova Semi-Circle Demo.

Connects to the demo MCP server and instructs Gemini to move the robot
along a semi-circle path defined by 3 calibration points.

Ref: https://github.com/googleapis/python-genai
"""
import asyncio
import os
import sys
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from google import genai
from google.genai import types, errors
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SERVER_URL = "http://127.0.0.1:8000/mcp"
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Calibration Data (P0, P90, P180) - HARDCODED as per requirements
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

async def send_message_with_retry(chat, message, max_retries=5, initial_delay=5.0):
    """Sends a message to Gemini with retry logic for rate limits (429)."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return await chat.send_message(message)
        except errors.ClientError as e:
            if e.code == 429:  # RESOURCE_EXHAUSTED
                print(f"Rate limit hit (429). Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                await asyncio.sleep(delay)
                delay *= 1.5  # Exponential backoff
            else:
                raise e
    raise RuntimeError(f"Max retries ({max_retries}) exceeded for Gemini API call.")

async def main():
    print(f"Connecting to MCP Server at {SERVER_URL}...")
    
    # Client for MCP Tools
    try:
        mcp_client = Client(StreamableHttpTransport(url=SERVER_URL))
    except Exception as e:
        print(f"Failed to connect to MCP server: {e}")
        return

    # Client for Gemini
    client = genai.Client(api_key=API_KEY)

    # Initialize simulation
    async with mcp_client:
        # Create a chat session
        chat = client.aio.chats.create(
            model="gemini-3-pro-preview", # Using flash for speed/cost, or pro if needed
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[mcp_client.session], # specific tool integration
                temperature=0.0, # Deterministic logic
            )
        )

        user_msg = "Begin the semi-circle demonstration."
        print(f"User: {user_msg}")

        # Turn 1: Send user message
        response = await send_message_with_retry(chat, user_msg)
        
        # Loop for tool calls (Manual/Half-auto loop to print progress)
        # We need to handle the turn-taking: Model calls tool -> We execute -> We send result -> Model continues
        
        while True:
            # Check if response has tool calls
            tool_calls = response.function_calls
            if not tool_calls:
                # No more tools, model is done (or asking question)
                print(f"Model: {response.text}")
                break

            # Execute all tool calls in this turn (usually 1 for sequential constraint, but API supports parallel)
            # We construct a list of FunctionResponse parts
            tool_outputs = []
            
            for call in tool_calls:
                fname = call.name
                fargs = call.args
                print(f" >> Tool Call: {fname}({fargs})")
                
                # Execute via MCP client
                try:
                    result = await mcp_client.call_tool(fname, **fargs)
                    output_text = str(result)
                except Exception as e:
                    output_text = f"Error: {str(e)}"
                
                print(f"    Result: {output_text}")
                
                tool_outputs.append(types.FunctionResponse(
                    name=fname,
                    response={"result": output_text} # structure depends on API expectation; dict usually safe
                ))

            # Send tool outputs back to model to continue generation
            response = await send_message_with_retry(chat, tool_outputs)

if __name__ == "__main__":
    asyncio.run(main())

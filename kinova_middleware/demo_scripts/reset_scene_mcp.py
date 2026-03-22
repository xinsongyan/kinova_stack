#!/usr/bin/env python3
"""
Simple script to reset the Kinova arm simulation scene via the FastMCP server.
"""
import asyncio
from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"

async def main():
    print(f"Connecting to FastMCP server at {SERVER_URL}...")
    try:
        async with Client(SERVER_URL) as client:
            print("Connected. Sending reset_scene command...")
            result = await client.call_tool("reset_scene")
            print(f"Result: {result.structured_content}")
            print("✓ Scene reset successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

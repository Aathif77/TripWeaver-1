from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any, Coroutine

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_FILES = {
    "hotel": PROJECT_ROOT / "mcp_servers" / "hotel_server.py",
    "flight": PROJECT_ROOT / "mcp_servers" / "flight_server.py",
}


def _decode_tool_result(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured

    content = getattr(result, "content", None) or []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"ok": True, "message": text}

    return {"ok": False, "error": "MCP tool returned no usable content."}


async def _call_tool_async(server: str, tool_name: str, arguments: dict[str, Any]) -> Any:
    server_path = SERVER_FILES[server]
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return _decode_tool_result(result)


def _run_in_new_thread(coro: Coroutine[Any, Any, Any]) -> Any:
    output: dict[str, Any] = {}

    def runner() -> None:
        try:
            output["value"] = asyncio.run(coro)
        except Exception as exc:  # converted into a recoverable tool response
            output["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=30)

    if thread.is_alive():
        return {"ok": False, "error": "MCP tool call timed out."}
    if "error" in output:
        return {"ok": False, "error": f"MCP tool call failed: {output['error']}"}
    return output.get("value", {"ok": False, "error": "MCP tool returned no result."})


def call_mcp_tool(server: str, tool_name: str, arguments: dict[str, Any]) -> Any:
    if server not in SERVER_FILES:
        return {"ok": False, "error": f"Unknown MCP server: {server}"}
    return _run_in_new_thread(_call_tool_async(server, tool_name, arguments))

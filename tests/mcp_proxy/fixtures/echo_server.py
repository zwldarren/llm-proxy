"""Minimal stdio MCP server used by the MCP proxy tests.

Exposes a single ``echo`` tool that returns its ``text`` argument unchanged,
plus (empty) prompts/resources lists. Run as a subprocess via the stdio
transport; it speaks JSON-RPC over stdin/stdout.
"""

import asyncio

from mcp.server import MCPServer

server = MCPServer("echo-test")


@server.tool()
async def echo(text: str) -> str:
    return text


async def main() -> None:
    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())

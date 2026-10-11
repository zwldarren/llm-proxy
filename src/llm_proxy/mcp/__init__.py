"""MCP proxy management package.

Public names are PEP 810 lazy imports, so `import llm_proxy.mcp` no longer pulls
the MCP SDK (and uvicorn/sse-starlette through it) into the startup graph; the
first touch of a name — normally `MCPProxyManager()` in the app lifespan — pays.
"""

lazy from llm_proxy.mcp.backend import BackendConnection, HTTPBackend, StdioBackend
lazy from llm_proxy.mcp.manager import MCPProxyManager
lazy from llm_proxy.mcp.proxy import MCPServerProxy

__all__ = [
    "BackendConnection",
    "HTTPBackend",
    "MCPProxyManager",
    "MCPServerProxy",
    "StdioBackend",
]

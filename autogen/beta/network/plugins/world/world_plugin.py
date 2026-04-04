# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""WorldPlugin — drop-in Hub plugin that serves a 3D network visualization.

Add to any Hub to get a live, interactive, low-poly 3D view of agents,
delegations, tool calls, and model responses in the browser.

Usage::

    from autogen.beta.network.plugins.world import WorldPlugin

    hub = Hub(plugins=[WorldPlugin(port=9000)])
    # Open http://localhost:9000 in browser
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

from aiohttp import web

from autogen.beta.events import (
    ModelResponse,
    ToolCallEvent,
    ToolResultEvent,
)
from autogen.beta.events.conditions import TypeCondition
from autogen.beta.network.events import (
    DelegationError,
    DelegationRejected,
    DelegationRequest,
    DelegationResult,
    TopicMessage,
)
from autogen.beta.network.topology import BasePlugin
from autogen.beta.stream import MemoryStream

_HERE = Path(__file__).parent

# Agent-level events forwarded to the browser
_AGENT_EVENT_TYPES = (ToolCallEvent, ToolResultEvent, ModelResponse)

# Color palette for islands
_PALETTE = [
    "#8B7355",
    "#C06050",
    "#5070A0",
    "#D08030",
    "#6A8060",
    "#8060A0",
    "#A07050",
    "#507080",
]


class WorldPlugin(BasePlugin):
    """Hub plugin that serves a 3D browser visualization of the network.

    On ``install``, subscribes to ``hub.stream`` for delegation events and
    wraps ``hub.ask`` / ``hub._delegate`` to inject per-call agent stream
    subscriptions so tool calls and model responses are also captured.

    Serves a single-page Three.js frontend on the configured port with a
    WebSocket that streams all events to connected browsers in real time.

    Args:
        port: HTTP port for the visualizer (default 9000).
        host: Bind address (default ``"0.0.0.0"``).
    """

    def __init__(self, port: int = 9000, host: str = "0.0.0.0") -> None:
        self._port = port
        self._host = host
        self._hub: Any = None
        self._sub_ids: list[int] = []
        self._clients: set[web.WebSocketResponse] = set()
        self._server_runner: web.AppRunner | None = None
        self._server_site: web.TCPSite | None = None
        self._original_ask: Any = None
        self._original_delegate: Any = None

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------

    def install(self, hub: Any) -> None:
        """Subscribe to hub events and start the HTTP/WS server."""
        self._hub = hub

        self._sub_ids.append(
            hub.stream.subscribe(
                self._on_hub_event,
                condition=TypeCondition((
                    DelegationRequest,
                    DelegationResult,
                    DelegationError,
                    DelegationRejected,
                    TopicMessage,
                )),
            )
        )

        # Wrap hub.ask and hub._delegate to inject agent stream taps
        self._original_ask = hub.ask
        self._original_delegate = hub._delegate
        hub.ask = self._wrapped_ask
        hub._delegate = self._wrapped_delegate

        asyncio.ensure_future(self._start_server())

    def uninstall(self) -> None:
        """Unsubscribe, restore original methods, stop server."""
        if self._hub:
            for sub_id in self._sub_ids:
                self._hub.stream.unsubscribe(sub_id)
            if self._original_ask:
                self._hub.ask = self._original_ask
            if self._original_delegate:
                self._hub._delegate = self._original_delegate
        self._sub_ids.clear()
        self._hub = None
        asyncio.ensure_future(self._stop_server())

    # ------------------------------------------------------------------
    # Hub event handler
    # ------------------------------------------------------------------

    async def _on_hub_event(self, event: Any) -> None:
        payload: dict[str, Any] | None = None

        if isinstance(event, DelegationRequest):
            payload = {"type": "DelegationRequest", "source": event.source, "target": event.target, "task": event.task}
        elif isinstance(event, DelegationResult):
            payload = {
                "type": "DelegationResult",
                "source": event.source,
                "target": event.target,
                "result": (event.result or "")[:200],
            }
        elif isinstance(event, DelegationRejected):
            payload = {
                "type": "DelegationRejected",
                "source": event.source,
                "target": event.target,
                "reason": event.reason,
            }
        elif isinstance(event, DelegationError):
            payload = {
                "type": "DelegationError",
                "source": event.source,
                "target": event.target,
                "error": event.error[:200],
            }
        elif isinstance(event, TopicMessage):
            payload = {
                "type": "TopicMessage",
                "topic": event.topic,
                "sender": event.sender,
                "message": event.message[:200],
            }

        if payload:
            await self._broadcast(payload)

    # ------------------------------------------------------------------
    # Agent stream tap
    # ------------------------------------------------------------------

    def _make_agent_stream_tap(self, agent_name: str) -> MemoryStream:
        """Create a stream that forwards agent events to the browser."""
        stream = MemoryStream()

        async def _on_agent_event(event: Any) -> None:
            if isinstance(event, ToolCallEvent):
                args_str = ""
                try:
                    args = event.serialized_arguments
                    parts = [f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in args.items()]
                    args_str = ", ".join(parts)[:120]
                except Exception:
                    args_str = (event.arguments or "")[:120]
                await self._broadcast({
                    "type": "ToolCall",
                    "agent": agent_name,
                    "tool": event.name,
                    "args": args_str,
                    "id": event.id,
                })
            elif isinstance(event, ToolResultEvent):
                await self._broadcast({
                    "type": "ToolResult",
                    "agent": agent_name,
                    "tool": event.name,
                    "result": (event.content or "")[:200],
                    "callId": event.parent_id,
                })
            elif isinstance(event, ModelResponse) and event.content:
                await self._broadcast({
                    "type": "ModelResponse",
                    "agent": agent_name,
                    "content": event.content[:200],
                })

        stream.subscribe(_on_agent_event, condition=TypeCondition(_AGENT_EVENT_TYPES))
        return stream

    async def _wrapped_ask(self, agent: Any, message: str, **kwargs: Any) -> Any:
        """Wraps ``hub.ask`` to inject an agent stream tap."""
        name = agent.name if hasattr(agent, "name") else str(agent)
        kwargs.setdefault("stream", self._make_agent_stream_tap(name))
        return await self._original_ask(agent, message, **kwargs)

    async def _wrapped_delegate(self, to_agent: str, task: str, **kwargs: Any) -> str:
        """Wraps ``hub._delegate`` to inject agent stream taps."""
        kwargs.setdefault("stream", self._make_agent_stream_tap(to_agent))
        return await self._original_delegate(to_agent, task, **kwargs)

    # ------------------------------------------------------------------
    # WebSocket broadcast
    # ------------------------------------------------------------------

    async def _broadcast(self, event: dict[str, Any]) -> None:
        if not self._clients:
            return
        msg = json.dumps(event)
        dead: set[web.WebSocketResponse] = set()
        for ws in self._clients:
            try:
                await ws.send_str(msg)
            except Exception:
                dead.add(ws)
        self._clients.difference_update(dead)

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------

    async def _start_server(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_ws)
        self._server_runner = web.AppRunner(app)
        await self._server_runner.setup()
        self._server_site = web.TCPSite(self._server_runner, self._host, self._port)
        await self._server_site.start()
        print(f"\n  World Visualizer: http://localhost:{self._port}\n")

    async def _stop_server(self) -> None:
        if self._server_site:
            await self._server_site.stop()
        if self._server_runner:
            await self._server_runner.cleanup()

    async def _handle_index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(_HERE / "index.html")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps(self._build_init_state()))
        self._clients.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            self._clients.discard(ws)
        return ws

    # ------------------------------------------------------------------
    # World state from Hub registry
    # ------------------------------------------------------------------

    def _build_init_state(self) -> dict[str, Any]:
        """Build init event from current Hub state."""
        if not self._hub:
            return {"type": "init", "islands": [], "agents": [], "bridges": []}

        registered = list(self._hub._agents.keys())
        n = max(1, len(registered))
        radius = 3.5 if n <= 4 else 2.5 * n / math.pi

        islands = []
        agents = []
        for i, name in enumerate(registered):
            angle = (i / n) * math.pi * 2 - math.pi / 2
            color = _PALETTE[i % len(_PALETTE)]
            label = name.replace("-", " ").replace("_", " ").title()
            islands.append({
                "id": name,
                "label": label,
                "x": round(math.cos(angle) * radius, 1),
                "y": round(0.2 * (i % 3), 1),
                "z": round(math.sin(angle) * radius, 1),
                "color": color,
                "visible": True,
            })
            agents.append({
                "id": name,
                "island": name,
                "label": label,
                "color": color,
                "capabilities": [],
            })

        # Connect neighbours and skip-one neighbours in the circle
        bridges = []
        for i in range(n):
            for step in (1, 2):
                j = (i + step) % n
                if j != i:
                    bridges.append({"from": registered[i], "to": registered[j], "visible": True})

        return {"type": "init", "islands": islands, "agents": agents, "bridges": bridges}

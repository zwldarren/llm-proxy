"""Unit tests for the Realtime bidirectional relay."""

import asyncio

from websockets import ConnectionClosed

from llm_proxy.realtime.relay import (
    RealtimeRelay,
    RealtimeRelayConfig,
    StarletteWebSocketAdapter,
    WebsocketsClientAdapter,
)


class FakeSocket:
    """Duck-typed client socket: scripted incoming queue, recorded sends.

    Emulates the Starlette WebSocket receive dict shape the
    StarletteWebSocketAdapter expects.
    """

    def __init__(self, incoming=None):
        self.incoming = list(incoming or [])
        self.sent: list[tuple[str, str | bytes]] = []
        self.close_code: int | None = None
        self.closed_with: int | None = None

    async def receive(self):
        if self.incoming:
            kind, data = self.incoming.pop(0)
            if kind == "disconnect":
                return {"type": "websocket.disconnect", "code": data}
            if kind == "text":
                return {"type": "websocket.receive", "text": data}
            return {"type": "websocket.receive", "bytes": data}
        # Block forever until cancelled (the relay cancels the other pump).
        await asyncio.Event().wait()

    async def send_text(self, data):
        self.sent.append(("text", data))

    async def send_bytes(self, data):
        self.sent.append(("binary", data))

    async def close(self, code):
        self.closed_with = code


class FakeUpstreamSocket(FakeSocket):
    """Upstream socket that raises ConnectionClosed when its queue is empty.

    Emulates the websockets ClientConnection interface the
    WebsocketsClientAdapter expects (``recv``/``send``, not ``receive``).
    """

    async def recv(self):
        if self.incoming:
            _, data = self.incoming.pop(0)
            return data
        raise ConnectionClosed(None, None)

    async def send(self, data):
        self.sent.append(data)


def _relay(
    client,
    upstream,
    *,
    config=None,
    on_upstream_message=None,
    on_client_error=None,
):
    return RealtimeRelay(
        client=StarletteWebSocketAdapter(client),
        upstream=WebsocketsClientAdapter(upstream),
        config=config,
        on_upstream_message=on_upstream_message,
        on_client_error=on_client_error,
    )


class TestRealtimeRelay:
    async def test_roundtrip_forwards_both_directions(self):
        """Messages flow both ways; upstream close propagates to the client."""
        client = FakeSocket(incoming=[("text", "client-msg-1"), ("binary", b"\x00\x01")])
        upstream = FakeUpstreamSocket(incoming=[("text", "server-msg-1"), ("text", "server-msg-2")])
        relay = _relay(client, upstream)
        await relay.run()

        assert upstream.sent == ["client-msg-1", b"\x00\x01"]
        assert client.sent == [("text", "server-msg-1"), ("text", "server-msg-2")]
        # Upstream closed abnormally (no close frame) → client closed with
        # the mapped 1011 (1006 must never be sent in a close frame).
        assert client.closed_with == 1011
        assert upstream.closed_with == 1011

    async def test_client_disconnect_propagates_to_upstream(self):
        """Client close code is propagated to the upstream."""
        client = FakeSocket(incoming=[("text", "bye"), ("disconnect", 1001)])
        upstream = FakeUpstreamSocket(incoming=[])
        relay = _relay(client, upstream)
        await relay.run()

        assert upstream.sent == ["bye"]
        assert upstream.closed_with == 1001

    async def test_client_abnormal_disconnect_propagates_mapped_code(self):
        """A reserved client code (1006) is mapped to 1011 before propagation."""
        client = FakeSocket(incoming=[("disconnect", 1006)])
        upstream = FakeUpstreamSocket(incoming=[])
        relay = _relay(client, upstream)
        await relay.run()

        # 1006 must never appear in a sent close frame (RFC 6455).
        assert upstream.closed_with == 1011

    async def test_oversized_message_rejected_but_connection_continues(self):
        """Oversized client messages trigger the error hook and are dropped."""
        errors = []

        async def record_error(code, message):
            errors.append((code, message))

        client = FakeSocket(
            incoming=[
                ("text", "x" * 10),
                ("text", "ok"),
            ]
        )
        upstream = FakeUpstreamSocket(incoming=[("text", "server-msg")])
        relay = _relay(
            client,
            upstream,
            config=RealtimeRelayConfig(max_message_bytes=5),
            on_client_error=record_error,
        )
        await relay.run()

        assert errors == [("invalid_request", "WebSocket message exceeds the 64 MiB size limit.")]
        assert upstream.sent == ["ok"]
        assert client.sent == [("text", "server-msg")]

    async def test_connection_age_cap(self):
        """The age cap triggers the error hook and closes both sides."""
        errors = []

        async def record_error(code, message):
            errors.append((code, message))

        client = FakeSocket(incoming=[])
        upstream = FakeUpstreamSocket(incoming=[])
        relay = _relay(
            client,
            upstream,
            config=RealtimeRelayConfig(max_connection_seconds=0),
            on_client_error=record_error,
        )
        await relay.run()

        assert errors == [
            (
                "websocket_connection_limit_reached",
                "WebSocket connection limit of 60 minutes reached.",
            )
        ]
        assert upstream.closed_with is not None
        assert client.closed_with is not None

    async def test_connection_age_cap_times_out_idle_receive(self):
        """A positive cap times out an idle client receive, not just the pre-check."""
        errors = []

        async def record_error(code, message):
            errors.append((code, message))

        class IdleUpstream:
            """Upstream that never closes on its own; blocks on recv until cancelled."""

            def __init__(self):
                self.sent: list[str | bytes] = []
                self.closed_with: int | None = None

            async def recv(self):
                await asyncio.Event().wait()

            async def send(self, data):
                self.sent.append(data)

            async def close(self, code):
                self.closed_with = code

        client = FakeSocket(incoming=[])
        upstream = IdleUpstream()
        relay = _relay(
            client,
            upstream,
            config=RealtimeRelayConfig(max_connection_seconds=0.05),
            on_client_error=record_error,
        )
        await relay.run()

        assert errors == [
            (
                "websocket_connection_limit_reached",
                "WebSocket connection limit of 60 minutes reached.",
            )
        ]
        assert upstream.closed_with is not None

    async def test_observation_hook_receives_upstream_messages(self):
        """The observation hook sees every upstream message; its failures are swallowed."""
        seen = []

        async def hook(kind, data):
            seen.append((kind, data))
            if data == "boom":
                raise RuntimeError("observer failure")

        client = FakeSocket(incoming=[])
        upstream = FakeUpstreamSocket(incoming=[("text", "first"), ("text", "boom")])
        relay = _relay(client, upstream, on_upstream_message=hook)
        await relay.run()

        assert seen == [("text", "first"), ("text", "boom")]
        # The hook failure did not break the relay: both messages were delivered.
        assert client.sent == [("text", "first"), ("text", "boom")]

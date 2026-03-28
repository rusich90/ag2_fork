# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections import defaultdict
from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio

from autogen.beta.context import Context
from autogen.beta.events import ModelMessage, ModelRequest, ToolCallEvent
from autogen.beta.streams.redis.serializer import Serializer

try:
    import redis  # noqa: F401

    _redis_installed = True
except ImportError:
    _redis_installed = False

pytestmark = [
    pytest.mark.redis,
    pytest.mark.asyncio,
    pytest.mark.skipif(not _redis_installed, reason="redis package not installed"),
]

REDIS_URL = "redis://mocked"


class MockRedis:
    """In-memory fake of redis.asyncio.Redis for testing."""

    def __init__(self) -> None:
        self._data: dict[str, list[bytes]] = defaultdict(list)
        self._pubsub_channels: dict[str, list[MockPubSub]] = defaultdict(list)

    async def rpush(self, key: str, value: bytes) -> None:
        self._data[key].append(value)

    async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
        if end == -1:
            return list(self._data[key][start:])
        return list(self._data[key][start : end + 1])

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def publish(self, channel: str, message: bytes) -> int:
        subs = self._pubsub_channels.get(channel, [])
        for ps in subs:
            await ps._receive(message)
        return len(subs)

    def pipeline(self, transaction: bool = True) -> "MockPipeline":
        return MockPipeline(self)

    def pubsub(self) -> "MockPubSub":
        return MockPubSub(self)

    async def aclose(self) -> None:
        pass


class MockPipeline:
    def __init__(self, redis: MockRedis) -> None:
        self._redis = redis
        self._ops: list[tuple[str, tuple]] = []

    async def __aenter__(self) -> "MockPipeline":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def delete(self, key: str) -> None:
        self._ops.append(("delete", (key,)))

    def rpush(self, key: str, value: bytes) -> None:
        self._ops.append(("rpush", (key, value)))

    async def execute(self) -> None:
        for op, args in self._ops:
            await getattr(self._redis, op)(*args)
        self._ops.clear()


class MockPubSub:
    def __init__(self, redis: MockRedis) -> None:
        self._redis = redis
        self._channels: list[str] = []
        self._queue: asyncio.Queue[dict] = asyncio.Queue()

    async def subscribe(self, channel: str) -> None:
        self._channels.append(channel)
        self._redis._pubsub_channels[channel].append(self)
        await self._queue.put({"type": "subscribe", "data": 1, "channel": channel})

    async def unsubscribe(self, channel: str) -> None:
        if channel in self._channels:
            self._channels.remove(channel)
            self._redis._pubsub_channels[channel].remove(self)

    async def _receive(self, data: bytes) -> None:
        await self._queue.put({"type": "message", "data": data})

    async def listen(self):  # noqa: ANN201
        while True:
            msg = await self._queue.get()
            yield msg

    async def aclose(self) -> None:
        pass


# Shared fake Redis instance per test — all RedisStream instances in the same
# test connect to the same "server" so cross-instance pub/sub works.
@pytest.fixture()
def mock_redis():
    return MockRedis()


@pytest_asyncio.fixture(params=[Serializer.JSON, Serializer.PICKLE], ids=["json", "pickle"])
async def redis_stream(mock_redis, request):
    from autogen.beta.streams.redis import RedisStream

    serializer = request.param
    streams: list = []

    def _make(**kwargs):
        kwargs.setdefault("prefix", f"ag2:test:{uuid4()}")
        kwargs.setdefault("serializer", serializer)
        with (
            patch("autogen.beta.streams.redis.stream.aioredis", create=True) as mock_aioredis,
            patch("autogen.beta.streams.redis.storage.aioredis", create=True) as mock_storage_aioredis,
        ):
            mock_aioredis.from_url.return_value = mock_redis
            mock_storage_aioredis.from_url.return_value = mock_redis
            s = RedisStream(REDIS_URL, **kwargs)
        streams.append(s)
        return s

    yield _make

    for s in streams:
        await s.close()


@pytest.fixture(params=[Serializer.JSON, Serializer.PICKLE], ids=["json", "pickle"])
def redis_storage(mock_redis, request):
    from autogen.beta.streams.redis import RedisStorage

    serializer = request.param

    def _make(prefix=None):
        with patch("autogen.beta.streams.redis.storage.aioredis", create=True) as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_redis
            return RedisStorage(REDIS_URL, prefix=prefix or f"ag2:test:{uuid4()}", serializer=serializer)

    return _make


@pytest.fixture()
def stream_id():
    return uuid4()


class TestRedisStorage:
    async def test_save_and_get_history(self, redis_storage, stream_id):
        storage = redis_storage()
        from autogen.beta.stream import MemoryStream

        stream = MemoryStream(storage, id=stream_id)
        ctx = Context(stream)

        event = ToolCallEvent(name="func1", arguments="test")
        await storage.save_event(event, ctx)

        history = list(await storage.get_history(stream_id))
        assert len(history) == 1
        assert history[0].name == "func1"
        assert history[0].arguments == "test"

    async def test_set_history(self, redis_storage, stream_id):
        storage = redis_storage()

        events = [
            ToolCallEvent(name="func1", arguments="a"),
            ToolCallEvent(name="func2", arguments="b"),
        ]
        await storage.set_history(stream_id, events)

        history = list(await storage.get_history(stream_id))
        assert len(history) == 2
        assert history[0].name == "func1"
        assert history[1].name == "func2"

    async def test_set_history_replaces(self, redis_storage, stream_id):
        storage = redis_storage()

        await storage.set_history(stream_id, [ToolCallEvent(name="old", arguments="x")])
        await storage.set_history(stream_id, [ToolCallEvent(name="new", arguments="y")])

        history = list(await storage.get_history(stream_id))
        assert len(history) == 1
        assert history[0].name == "new"

    async def test_drop_history(self, redis_storage, stream_id):
        storage = redis_storage()

        await storage.set_history(stream_id, [ToolCallEvent(name="f", arguments="a")])
        await storage.drop_history(stream_id)

        history = list(await storage.get_history(stream_id))
        assert len(history) == 0

    async def test_empty_history(self, redis_storage, stream_id):
        storage = redis_storage()
        history = list(await storage.get_history(stream_id))
        assert history == []


class TestRedisStream:
    async def test_send_event_persists(self, redis_stream):
        stream = redis_stream()
        event = ToolCallEvent(name="func1", arguments="test")
        await stream.send(event, context=Context(stream))

        history = list(await stream.history.get_events())
        assert len(history) == 1
        assert history[0].name == "func1"

    async def test_send_notifies_subscribers(self, redis_stream):
        stream = redis_stream()
        received = []

        stream.subscribe(lambda ev: received.append(ev))
        stream._ensure_listener()

        event = ToolCallEvent(name="func1", arguments="test")
        await stream.send(event, context=Context(stream))

        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].name == "func1"

    async def test_send_multiple_events(self, redis_stream):
        stream = redis_stream()
        ctx = Context(stream)

        event1 = ToolCallEvent(name="func1", arguments="a")
        event2 = ModelMessage(content="hello")

        await stream.send(event1, context=ctx)
        await stream.send(event2, context=ctx)

        history = list(await stream.history.get_events())
        assert len(history) == 2

    async def test_history_persists_across_instances(self, redis_stream):
        prefix = f"ag2:test:{uuid4()}"
        sid = uuid4()

        stream1 = redis_stream(prefix=prefix, id=sid)
        await stream1.send(ToolCallEvent(name="func1", arguments="a"), context=Context(stream1))

        # New stream instance, same id and prefix — shares the same fake redis
        stream2 = redis_stream(prefix=prefix, id=sid)
        history = list(await stream2.history.get_events())
        assert len(history) == 1
        assert history[0].name == "func1"

    async def test_no_duplicate_persistence(self, redis_stream):
        """Events should be persisted exactly once, not duplicated."""
        prefix = f"ag2:test:{uuid4()}"
        sid = uuid4()

        stream1 = redis_stream(prefix=prefix, id=sid)
        stream2 = redis_stream(prefix=prefix, id=sid)
        stream2._ensure_listener()
        await asyncio.sleep(0.1)

        await stream1.send(ToolCallEvent(name="func1", arguments="a"), context=Context(stream1))
        await asyncio.sleep(0.1)

        history = list(await stream1.history.get_events())
        assert len(history) == 1

    async def test_cross_instance_pubsub(self, redis_stream):
        """Events sent on one instance are received by subscribers on another."""
        prefix = f"ag2:test:{uuid4()}"
        sid = uuid4()

        stream1 = redis_stream(prefix=prefix, id=sid)
        stream2 = redis_stream(prefix=prefix, id=sid)

        received = []
        stream2.subscribe(lambda ev: received.append(ev))
        stream2._ensure_listener()
        await asyncio.sleep(0.1)

        event = ToolCallEvent(name="cross_process", arguments="hello")
        await stream1.send(event, context=Context(stream1))

        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].name == "cross_process"

    async def test_bidirectional_pubsub(self, redis_stream):
        """Both instances can send and receive from each other."""
        prefix = f"ag2:test:{uuid4()}"
        sid = uuid4()

        stream1 = redis_stream(prefix=prefix, id=sid)
        stream2 = redis_stream(prefix=prefix, id=sid)

        received1 = []
        received2 = []
        stream1.subscribe(lambda ev: received1.append(ev))
        stream2.subscribe(lambda ev: received2.append(ev))
        stream1._ensure_listener()
        stream2._ensure_listener()
        await asyncio.sleep(0.1)

        await stream1.send(ModelRequest(content="from stream1"), context=Context(stream1))
        await asyncio.sleep(0.1)

        await stream2.send(ModelMessage(content="from stream2"), context=Context(stream2))
        await asyncio.sleep(0.1)

        # stream1 receives both (own + stream2's)
        assert len(received1) == 2
        # stream2 receives both (stream1's + own)
        assert len(received2) == 2

        history = list(await stream1.history.get_events())
        assert len(history) == 2

    async def test_where_filter_with_pubsub(self, redis_stream):
        """Filtered subscriptions work correctly with pub/sub delivery."""
        stream = redis_stream()
        tool_events = []

        stream.where(ToolCallEvent).subscribe(lambda ev: tool_events.append(ev))
        stream._ensure_listener()

        await stream.send(ToolCallEvent(name="func1", arguments="a"), context=Context(stream))
        await stream.send(ModelMessage(content="hello"), context=Context(stream))
        await asyncio.sleep(0.1)

        assert len(tool_events) == 1
        assert tool_events[0].name == "func1"

    async def test_reentrant_send_no_deadlock(self, redis_stream):
        """Subscriber that sends an event and waits for the response must not deadlock.

        tool-call deadlock pattern flow:
        1. Subscriber receives ModelRequest
        2. Subscriber sends ModelMessage and waits for a ToolCallEvent response
        3. Another subscriber handles ModelMessage by sending ToolCallEvent
        4. If dispatch is serialized through a single listener task, step 3
           can't fire until step 2 finishes — but step 2 is waiting for step 3.

        With local dispatch (MemoryStream.send), step 3 executes inline
        during step 2's send() call, so the response arrives immediately.
        """
        stream = redis_stream()
        ctx = Context(stream)
        response_received = asyncio.Event()
        tool_result = []

        async def client_subscriber(ev):
            """Simulates _call_client: sends a request and waits for response."""
            if isinstance(ev, ModelRequest):
                # Send response and wait for the tool result to come back
                await stream.send(ModelMessage(content="need_tool"), ctx)
                # Wait for the ToolCallEvent that tool_executor will send
                await asyncio.wait_for(response_received.wait(), timeout=3)

        async def tool_executor(ev):
            """Simulates tool executor: fires tool result when it sees ModelMessage."""
            if isinstance(ev, ModelMessage) and ev.content == "need_tool":
                await stream.send(ToolCallEvent(name="greet", arguments="Mark"), ctx)

        async def result_collector(ev):
            """Signals when the tool result arrives."""
            if isinstance(ev, ToolCallEvent):
                tool_result.append(ev)
                response_received.set()

        stream.subscribe(client_subscriber)
        stream.subscribe(tool_executor)
        stream.subscribe(result_collector)

        # This should complete without deadlock
        await asyncio.wait_for(
            stream.send(ModelRequest(content="request"), ctx),
            timeout=5,
        )
        await asyncio.sleep(0.2)

        # The tool result was received — no deadlock
        assert len(tool_result) == 1
        assert tool_result[0].name == "greet"

    async def test_deeply_nested_send(self, redis_stream):
        """Three levels of nested send() with waits — simulates tool call chain."""
        stream = redis_stream()
        ctx = Context(stream)
        received = []
        tool_done = asyncio.Event()
        final_done = asyncio.Event()

        async def level1(ev):
            if isinstance(ev, ModelRequest):
                received.append(("L1", ev))
                await stream.send(ToolCallEvent(name="tool1", arguments="a"), ctx)
                await asyncio.wait_for(tool_done.wait(), timeout=3)

        async def level2(ev):
            if isinstance(ev, ToolCallEvent):
                received.append(("L2", ev))
                await stream.send(ModelMessage(content="done"), ctx)
                tool_done.set()

        async def level3(ev):
            if isinstance(ev, ModelMessage) and ev.content == "done":
                received.append(("L3", ev))
                final_done.set()

        stream.subscribe(level1)
        stream.subscribe(level2)
        stream.subscribe(level3)

        await asyncio.wait_for(
            stream.send(ModelRequest(content="start"), ctx),
            timeout=5,
        )
        await asyncio.wait_for(final_done.wait(), timeout=5)

        assert len(received) == 3
        assert received[0][0] == "L1"
        assert received[1][0] == "L2"
        assert received[2][0] == "L3"

    async def test_multiple_subscribers_same_instance(self, redis_stream):
        """Multiple subscribers on the same stream all receive events."""
        stream = redis_stream()
        received_a = []
        received_b = []
        received_c = []

        stream.subscribe(lambda ev: received_a.append(ev))
        stream.subscribe(lambda ev: received_b.append(ev))
        stream.subscribe(lambda ev: received_c.append(ev))
        stream._ensure_listener()

        await stream.send(ToolCallEvent(name="func1", arguments="a"), context=Context(stream))
        await asyncio.sleep(0.1)

        assert len(received_a) == 1
        assert len(received_b) == 1
        assert len(received_c) == 1

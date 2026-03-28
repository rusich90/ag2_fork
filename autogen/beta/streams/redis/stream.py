# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import json
from uuid import uuid4

from autogen.beta.context import Context, StreamId
from autogen.beta.events import BaseEvent
from autogen.beta.stream import MemoryStream
from autogen.import_utils import optional_import_block

from .serializer import Serializer, deserialize, serialize
from .storage import RedisStorage

with optional_import_block():
    import redis.asyncio as aioredis


class RedisStream(MemoryStream):
    """A full-featured stream with Redis-backed pub/sub and persistent event history.

    All events flow through Redis Pub/Sub, ensuring subscribers across processes
    and machines receive every event. History is persisted to Redis.

    Event flow (local):
        send() → dispatch to local subscribers + persist to Redis + publish to Pub/Sub

    Event flow (cross-process):
        listener → receives from Pub/Sub → skips own messages → dispatches to local subscribers

    Args:
        redis_url: Redis connection URL.
        prefix: Key prefix for Redis storage and pub/sub channels.
        id: Stream ID. If None, a new UUID is generated.
        serializer: Serialization format (Serializer.JSON or Serializer.PICKLE).
    """

    def __init__(
        self,
        redis_url: str,
        *,
        prefix: str = "ag2:stream",
        id: StreamId | None = None,
        serializer: Serializer = Serializer.JSON,
    ) -> None:
        storage = RedisStorage(redis_url, prefix=prefix, serializer=serializer)
        super().__init__(storage, id=id)

        # Unsubscribe the auto-registered save_event from MemoryStream.__init__
        # We handle persistence explicitly in send() to avoid double-writes
        storage_sub_id = next(iter(self._subscribers))
        self.unsubscribe(storage_sub_id)

        self._redis_storage = storage
        self._redis_url = redis_url
        self._prefix = prefix
        self._serializer = serializer
        self._channel = f"{prefix}:pubsub:{self.id}"
        self._instance_id = str(uuid4())
        self._listener_task: asyncio.Task | None = None
        self._listener_ready = asyncio.Event()
        self._pubsub_redis = aioredis.from_url(redis_url)
        self._publish_redis = aioredis.from_url(redis_url)

    def _ensure_listener(self) -> None:
        """Start the Redis Pub/Sub listener if not already running."""
        if self._listener_task is None or self._listener_task.done():
            self._listener_ready.clear()
            self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        """Listen for events on the Redis Pub/Sub channel and dispatch to local subscribers.

        Skips messages published by this instance (already dispatched locally in send()).
        Only dispatches messages from other instances (cross-process events).
        """
        pubsub = self._pubsub_redis.pubsub()
        await pubsub.subscribe(self._channel)
        self._listener_ready.set()
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                # Extract sender instance_id from the envelope
                raw = message["data"]
                sender_id, event_data = self._unwrap_envelope(raw)

                # Skip events from this instance -- already dispatched in send()
                if sender_id == self._instance_id:
                    continue

                event = deserialize(event_data, self._serializer)
                if isinstance(event, BaseEvent):
                    await super().send(event, Context(self))
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(self._channel)
            await pubsub.aclose()

    def _wrap_envelope(self, event_bytes: bytes) -> bytes:
        """Wrap serialized event with instance_id for origin tracking."""
        envelope = {"instance_id": self._instance_id, "data": event_bytes.decode("latin-1")}
        return json.dumps(envelope).encode()

    def _unwrap_envelope(self, raw: bytes) -> tuple[str, bytes]:
        """Unwrap envelope to get (instance_id, event_bytes)."""
        try:
            envelope = json.loads(raw)
            return envelope["instance_id"], envelope["data"].encode("latin-1")
        except (json.JSONDecodeError, KeyError):
            # Legacy message without envelope -- treat as foreign
            return "", raw

    async def send(self, event: BaseEvent, context: Context) -> None:
        """Persist to Redis, dispatch locally, and publish for cross-process consumers."""
        self._ensure_listener()
        await self._listener_ready.wait()

        # Persist to Redis first so subscribers read history during dispatch
        await self._redis_storage.save_event(event, context)

        # Dispatch to local subscribers
        await super().send(event, context)

        # Publish to Redis pub/sub for cross-process listeners
        event_bytes = serialize(event, self._serializer)
        await self._publish_redis.publish(self._channel, self._wrap_envelope(event_bytes))

    async def close(self) -> None:
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
        await self._pubsub_redis.aclose()
        await self._publish_redis.aclose()
        await self._redis_storage.close()

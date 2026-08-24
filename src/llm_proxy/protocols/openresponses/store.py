"""Redis-backed storage for OpenResponses responses."""

from typing import Any

import orjson
import redis.asyncio as redis


class ResponseStore:
    """Redis-backed storage for OpenResponses responses.

    Stores complete responses for retrieval and conversation continuity.
    Uses Redis with TTL for automatic expiration. Responses are namespaced
    by ``api_key_name`` so one tenant cannot read or delete another tenant's
    stored responses.
    """

    def __init__(self, redis_client: redis.Redis, ttl: int = 86400) -> None:
        self.redis = redis_client
        self.ttl = ttl

    _KEY_DELIMITER = "|"

    def _key(self, api_key_name: str, response_id: str) -> str:
        return f"openresponses:response:{api_key_name}{self._KEY_DELIMITER}{response_id}"

    async def store(self, api_key_name: str, response_id: str, response: dict[str, Any]) -> None:
        key = self._key(api_key_name, response_id)
        await self.redis.setex(key, self.ttl, orjson.dumps(response))

    async def retrieve(self, api_key_name: str, response_id: str) -> dict[str, Any] | None:
        key = self._key(api_key_name, response_id)
        data = await self.redis.get(key)
        return orjson.loads(data) if data else None

    async def delete(self, api_key_name: str, response_id: str) -> bool:
        key = self._key(api_key_name, response_id)
        return await self.redis.delete(key) > 0

    async def exists(self, api_key_name: str, response_id: str) -> bool:
        key = self._key(api_key_name, response_id)
        return await self.redis.exists(key) > 0


__all__ = ["ResponseStore"]

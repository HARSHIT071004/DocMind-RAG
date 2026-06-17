import time
from collections import OrderedDict

from rag.config import settings


class QueryCache:
    def __init__(self, maxsize: int = 128, ttl: int = 300):
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    def get(self, key: str) -> dict | None:
        if key not in self._cache:
            return None
        ts, value = self._cache[key]
        if time.monotonic() - ts > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: dict) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.monotonic(), value)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


query_cache = QueryCache(maxsize=settings.CACHE_MAXSIZE, ttl=settings.CACHE_TTL_SECONDS)

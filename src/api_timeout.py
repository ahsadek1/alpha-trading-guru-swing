"""
src/api_timeout.py — Enforces hard timeouts on all external AI/market API calls.
Import and use wrap_with_timeout() to prevent any single API call from
blocking the asyncio event loop or trading scan loop indefinitely.
"""
import asyncio
import logging
import functools
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 8.0   # seconds — max any single API call may take
AI_TIMEOUT      = 10.0  # slightly more for AI inference calls
MARKET_TIMEOUT  = 6.0   # fast-fail for market data (Alpaca)


async def with_timeout(coro, seconds: float = DEFAULT_TIMEOUT, fallback: Any = None, label: str = ""):
    """
    Await a coroutine with a hard timeout.
    Returns fallback value on timeout instead of blocking forever.
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        logger.warning("API timeout after %.1fs%s — using fallback", seconds, f" [{label}]" if label else "")
        return fallback
    except Exception as e:
        logger.warning("API error%s: %s", f" [{label}]" if label else "", e)
        return fallback


def sync_timeout(seconds: float = DEFAULT_TIMEOUT, fallback: Any = None, label: str = ""):
    """
    Decorator: adds a thread-based timeout to any synchronous function.
    Prevents blocking scan loops on slow API calls.
    """
    import concurrent.futures
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    logger.warning("Sync timeout after %.1fs%s", seconds, f" [{label}]" if label else "")
                    return fallback
                except Exception as e:
                    logger.warning("Sync error%s: %s", f" [{label}]" if label else "", e)
                    return fallback
        return wrapper
    return decorator

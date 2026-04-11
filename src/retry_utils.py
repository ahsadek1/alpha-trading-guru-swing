"""
src/retry_utils.py — Retry decorator for external API calls.
Automatically retries on transient network errors with exponential backoff.
Prevents single flaky requests from blocking the trading loop.
"""
import time
import logging
import functools
from typing import Tuple, Type

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    backoff_base: float = 1.5,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_failure=None,
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Total attempts before giving up (default 3)
        backoff_base: Seconds for first retry; doubles each time (default 1.5s)
        exceptions:   Which exception types to retry on
        on_failure:   Optional callable(fn_name, attempt, exc) for custom logging
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_attempts:
                        logger.warning(
                            "%s failed after %d attempts: %s",
                            func.__name__, max_attempts, e
                        )
                        if on_failure:
                            on_failure(func.__name__, attempt, e)
                        raise
                    wait = backoff_base * (2 ** (attempt - 1))
                    logger.debug(
                        "%s attempt %d/%d failed (%s) — retrying in %.1fs",
                        func.__name__, attempt, max_attempts, e, wait
                    )
                    time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


def retry_async(
    max_attempts: int = 3,
    backoff_base: float = 1.5,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Async version of the retry decorator."""
    import asyncio

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_attempts:
                        logger.warning(
                            "%s async failed after %d attempts: %s",
                            func.__name__, max_attempts, e
                        )
                        raise
                    wait = backoff_base * (2 ** (attempt - 1))
                    logger.debug(
                        "%s async attempt %d/%d failed — retrying in %.1fs",
                        func.__name__, attempt, wait
                    )
                    await asyncio.sleep(wait)
            raise last_exc
        return wrapper
    return decorator

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


def log_time(func: Callable) -> Callable:
    """Decorator to log the execution time of a function.

    Parameters
    ----------
    func : callable
        The function whose execution time will be logged.

    Returns
    -------
    callable
        The wrapped function with logging of execution time.
    """
    logger = logging.getLogger(
        func.__module__
    )  # Get logger for the caller's module

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa
        logger.info(f"Start: {func.__name__}")
        start = time.process_time()
        result = func(*args, **kwargs)
        end = time.process_time()
        logger.info(f"End: {func.__name__} Execution time: {end - start:.6f} s")
        return result

    return wrapper

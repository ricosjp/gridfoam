"""Tests for log_time module."""

import logging
import time
from unittest.mock import patch

import pytest

from gridfoam.log_time import log_time


def test_log_time_decorator():
    """Test that log_time decorator logs execution time."""
    logger = logging.getLogger("gridfoam.log_time")

    @log_time
    def test_func(x: int, y: int) -> int:
        return x + y

    with patch.object(logger, "info") as mock_info, \
         patch("time.process_time", side_effect=[0.0, 0.001]):
        result = test_func(1, 2)
        assert result == 3
        assert mock_info.call_count == 2
        assert "Start: test_func" in str(mock_info.call_args_list[0])
        assert "End: test_func" in str(mock_info.call_args_list[1])


def test_log_time_preserves_function_metadata():
    """Test that log_time preserves function metadata."""
    @log_time
    def test_func(x: int, y: int) -> int:
        """Test function docstring."""
        return x + y

    assert test_func.__name__ == "test_func"
    assert test_func.__doc__ == "Test function docstring."

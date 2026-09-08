"""Iteration snapshots copy residual dictionaries and refuse in-place writes."""

from collections.abc import MutableMapping
from typing import cast

import pytest

from gridfoam.algorithms.iteration_state import IterationState


def test_residual_mappings_are_copied_and_read_only() -> None:
    """Caller dicts and default empty maps cannot mutate the snapshot."""
    initial = {"U": 1.0}
    current = {"U": 0.1}
    state = IterationState(3, initial, current, True)
    initial["U"] = 2.0
    current.clear()
    assert state.initial_residuals == {"U": 1.0}
    assert state.current_residuals == {"U": 0.1}
    default = IterationState(0)
    for residuals in (
        state.initial_residuals,
        state.current_residuals,
        default.initial_residuals,
        default.current_residuals,
    ):
        with pytest.raises(TypeError):
            cast(MutableMapping[str, float], residuals)["U"] = 9.0

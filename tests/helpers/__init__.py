"""Shared helpers for gridfoam test modules."""

from tests.helpers.configs import (
    channel_config,
    potential_flow_config,
    refined_3d_config,
    refined_config,
    simple_convergence_config,
)
from tests.helpers.fields import interior_mask, linear_scalar_field
from tests.helpers.grids import refined_3d_grid, refined_grid

__all__ = [
    "channel_config",
    "interior_mask",
    "linear_scalar_field",
    "potential_flow_config",
    "refined_3d_config",
    "refined_3d_grid",
    "refined_config",
    "refined_grid",
    "simple_convergence_config",
]

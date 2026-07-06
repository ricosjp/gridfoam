"""Grid factories built from shared test configurations."""

from __future__ import annotations

from tests.helpers.configs import refined_3d_config, refined_config

from gridfoam.core.grid.base import IGridBase
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.enums import GradScheme


def refined_grid(*, grad_scheme: GradScheme | None = None) -> IGridBase:
    """2-D refined axis-projected grid."""
    return create_grid(refined_config(grad_scheme=grad_scheme))


def refined_3d_grid(grad_scheme: GradScheme) -> IGridBase:
    """3-D refined axis-projected grid."""
    return create_grid(refined_3d_config(grad_scheme=grad_scheme))

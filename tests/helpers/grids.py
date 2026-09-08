"""Grid factories built from shared test configurations."""

from __future__ import annotations

from pathlib import Path

import pyvista as pv
from tests.conftest import small_gridfoam_config
from tests.helpers.configs import refined_3d_config, refined_config

from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import GridBase
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import DomainConfig, fvSchemesConfig
from gridfoam.meta.enums import GradScheme


def refined_grid(
    *,
    grad_scheme: GradScheme | None = None,
    fv_schemes: fvSchemesConfig | None = None,
) -> GridBase:
    """2-D refined axis-projected grid."""
    return create_grid(
        refined_config(grad_scheme=grad_scheme, fv_schemes=fv_schemes)
    )


def refined_3d_grid(
    grad_scheme: GradScheme | None = None,
    *,
    fv_schemes: fvSchemesConfig | None = None,
) -> GridBase:
    """3-D refined axis-projected grid."""
    return create_grid(
        refined_3d_config(grad_scheme=grad_scheme, fv_schemes=fv_schemes)
    )


def immersed_plane_grid(
    path: Path, n: int, theta: float, scale: float = 1.0
) -> AxisProjectedGrid:
    """Uniform grid cut by a plane at a given fractional cell distance."""
    path.mkdir(parents=True, exist_ok=True)
    interface = (0.5 - 0.5 / n + theta / n) * scale
    mesh_path = path / "plane.stl"
    pv.Plane(
        center=(interface, 0.05 * scale, 0.05 * scale),
        direction=(1, 0, 0),
        i_size=scale,
        j_size=scale,
    ).triangulate().save(mesh_path)
    config = small_gridfoam_config(output_dir=path)
    config = config.model_copy(
        update={
            "fluxel": config.fluxel.model_copy(
                update={
                    "domain": DomainConfig(
                        lower=[0, 0, 0], upper=[scale, 0.1 * scale, 0.1 * scale]
                    ),
                    "root_resolution": [n, 1, 1],
                    "mesh_path": mesh_path,
                }
            )
        }
    )
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)
    assert grid.num_immersed_faces == (2 if theta == 0 else 1)
    return grid

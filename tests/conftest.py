"""Shared pytest fixtures for gridfoam tests."""

from __future__ import annotations

import pathlib

import pytest

from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import (
    ControlConfig,
    DomainConfig,
    FluxelConfig,
    GridfoamConfig,
    OutputConfig,
    SimulatorConfig,
    SolverConfig,
    fvSchemesConfig,
    fvSolutionConfig,
)
from gridfoam.meta.enums import DeviceType, IbmType, PrecisionType, SolverType


def small_gridfoam_config(
    *,
    output_dir: pathlib.Path | None = None,
) -> GridfoamConfig:
    """
    Build a tiny axis-projected configuration for fast tests.

    Parameters
    ----------
    output_dir : pathlib.Path | None
        Optional output directory for control config. Defaults to a path
        under the system temp directory.
    """
    out = output_dir or pathlib.Path("/tmp/gridfoam_pytest_out")
    return GridfoamConfig(
        fluxel=FluxelConfig(
            domain=DomainConfig(
                lower=[0.0, 0.0, 0.0],
                upper=[1.0, 1.0, 0.1],
            ),
            root_resolution=[4, 4, 1],
            target_level=0,
            n_leaf_refinement=0,
            mesh_path=None,
            ibm_type=IbmType.AXIS_PROJECTED,
        ),
        simulator=SimulatorConfig(
            control=ControlConfig(
                deltaT=0.01,
                endTime=0.02,
                writeInterval=1,
                output=OutputConfig(
                    output_dir=out,
                    base_name="pytest",
                ),
                precision=PrecisionType.FLOAT64,
            ),
            fvSchemes=fvSchemesConfig(),
            fvSolution=fvSolutionConfig(
                solvers={
                    "p": SolverConfig(method=SolverType.CG),
                },
            ),
            boundaryConditions=None,
            device=DeviceType.CPU,
        ),
    )


@pytest.fixture(scope="session")
def small_axis_projected_grid() -> AxisProjectedGrid:
    """
    Session-scoped minimal axis-projected grid.

    Fluxel mesh build runs once per session.
    """
    config = small_gridfoam_config()
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)
    return grid

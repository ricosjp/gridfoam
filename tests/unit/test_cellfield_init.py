"""Unit tests for ``CellField`` initialization from config conditions."""

from __future__ import annotations

import pathlib

import pytest
import torch

from gridfoam.core.dimensions import DIM_KIN_PRESSURE, DIM_VELOCITY
from gridfoam.core.field import CellField
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import (
    ConditionConfig,
    ControlConfig,
    DomainConfig,
    FluxelConfig,
    GridfoamConfig,
    LaminarConfig,
    ManualAlgorithm,
    NewtonianTransportConfig,
    OutputConfig,
    PropertiesConfig,
    SimulatorConfig,
    SolverConfig,
    fvSchemesConfig,
    fvSolutionConfig,
)
from gridfoam.meta.enums import (
    AlgorithmType,
    DeviceType,
    FieldRole,
    IbmType,
    PrecisionType,
    SolverType,
    TransportModelType,
    TurbulenceType,
)


def _gridfoam_config_with_conditions() -> GridfoamConfig:
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
                    output_dir=pathlib.Path("/tmp/gridfoam_pytest_out"),
                    base_name="pytest",
                ),
                precision=PrecisionType.FLOAT64,
            ),
            fvSchemes=fvSchemesConfig(),
            fvSolution=fvSolutionConfig(
                algorithm=ManualAlgorithm(type=AlgorithmType.MANUAL),
                solvers={"p": SolverConfig(method=SolverType.CG)},
            ),
            conditions={
                "U": ConditionConfig(
                    internal=[1.5, 0.0, 0.0],
                    boundary={},
                ),
                "p": ConditionConfig(
                    internal=[2.5],
                    boundary={},
                ),
                "bad": ConditionConfig(
                    internal=[1.0],
                    boundary={},
                ),
            },
            properties=PropertiesConfig(
                transport=NewtonianTransportConfig(
                    type=TransportModelType.NEWTONIAN,
                    nu=0.1,
                ),
                turbulence=LaminarConfig(type=TurbulenceType.LAMINAR),
            ),
            device=DeviceType.CPU,
        ),
    )


def test_cellfield_initializes_from_conditions_internal() -> None:
    # Fields listed in config conditions must be filled from internal values.
    grid = create_grid(_gridfoam_config_with_conditions())
    U = CellField(grid, "U", FieldRole.LOCAL, num_components=3)
    p = CellField(grid, "p", FieldRole.LOCAL, num_components=1)

    expected_u = torch.tensor([1.5, 0.0, 0.0], dtype=grid.dtype)
    expected_p = torch.tensor([2.5], dtype=grid.dtype)
    torch.testing.assert_close(U.data, expected_u.expand_as(U.data))
    torch.testing.assert_close(p.data, expected_p.expand_as(p.data))
    assert U.dimension == DIM_VELOCITY
    assert p.dimension == DIM_KIN_PRESSURE


def test_cellfield_without_conditions_stays_zero() -> None:
    # Fields absent from conditions must start at zero.
    grid = create_grid(_gridfoam_config_with_conditions())
    field = CellField(grid, "k", FieldRole.LOCAL, num_components=1)

    torch.testing.assert_close(field.data, torch.zeros_like(field.data))


def test_cellfield_rejects_mismatched_internal_length() -> None:
    # Internal value count must match num_components.
    grid = create_grid(_gridfoam_config_with_conditions())

    with pytest.raises(ValueError, match="internal has 1 value"):
        CellField(grid, "bad", FieldRole.LOCAL, num_components=3)

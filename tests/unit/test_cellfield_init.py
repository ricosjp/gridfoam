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
                    internal=2.5,
                    boundary={},
                ),
                "bad": ConditionConfig(
                    internal=1.0,
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
    U = CellField(grid, "U", FieldRole.LOCAL, component_shape=(3,))
    p = CellField(grid, "p", FieldRole.LOCAL, component_shape=())

    expected_u = torch.tensor([1.5, 0.0, 0.0], dtype=grid.dtype)
    expected_p = torch.tensor(2.5, dtype=grid.dtype)
    torch.testing.assert_close(U.data, expected_u.expand_as(U.data))
    torch.testing.assert_close(p.data, expected_p.expand_as(p.data))
    assert U.dimension == DIM_VELOCITY
    assert p.dimension == DIM_KIN_PRESSURE


def test_cellfield_without_conditions_stays_zero() -> None:
    # Fields absent from conditions must start at zero.
    grid = create_grid(_gridfoam_config_with_conditions())
    field = CellField(grid, "k", FieldRole.LOCAL, component_shape=())

    torch.testing.assert_close(field.data, torch.zeros_like(field.data))


def test_cellfield_rejects_mismatched_internal_length() -> None:
    # Internal value count must match num_components.
    grid = create_grid(_gridfoam_config_with_conditions())

    with pytest.raises(ValueError, match="internal.*shape"):
        CellField(grid, "bad", FieldRole.LOCAL, component_shape=(3,))


def test_nested_tensor_config_and_vtu_export() -> None:
    from gridfoam.fv import fvc
    from gridfoam.io.vtu import to_unstructured_grid, update_export_cell_data
    from gridfoam.meta.config import ConditionConfig

    config = _gridfoam_config_with_conditions()
    value = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    tensor_condition = ConditionConfig.model_validate(
        {
            "internal": value,
            "export": True,
            "boundary": {
                "wall": {
                    "type": "dirichlet",
                    "patches": ["x_minus"],
                    "value": value,
                }
            },
        }
    )
    config = config.model_copy(
        update={
            "simulator": config.simulator.model_copy(
                update={"conditions": {"stress": tensor_condition}}
            )
        }
    )
    grid = create_grid(config)
    stress = CellField(grid, "stress", FieldRole.LOCAL, (3, 3))
    expected = torch.tensor(value, dtype=grid.dtype)
    torch.testing.assert_close(stress.data, expected.expand_as(stress.data))
    face = fvc.interpolate(stress)
    torch.testing.assert_close(
        face.domain_bnd_data, expected.expand_as(face.domain_bnd_data)
    )
    exported = to_unstructured_grid(grid)
    update_export_cell_data(exported, grid)
    assert exported.cell_data["stress"].shape == (grid.num_cells, 9)
    # Export's external representation must never replace the field data.
    assert stress.data.shape == (grid.num_cells, 3, 3)

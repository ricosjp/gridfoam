"""Tests for CTX/application/fractional_step module."""

import pathlib
from unittest.mock import MagicMock, create_autospec, patch

from gridfoam.CTX.context import SimulationContext
from gridfoam.DNA.config import fvSchemesConfig, fvSolutionConfig
from gridfoam.DNA.enum import NormType
from gridfoam.DNA.scheme.fvm.ddt._choice import FVMDdtSchemeChoice
from gridfoam.DNA.scheme.fvm.div._choice import FVMDivSchemeChoice
from gridfoam.DNA.scheme.solver._choice import SolverMethodChoice
from gridfoam.RNA.grid_handle import GridHandle
from gridfoam.RNA.registry import SimulationMetaRegistry


def test_fractional_step_engine_import():
    """Test that FractionalStepEngine can be imported."""
    from gridfoam.CTX.application.fractional_step import FractionalStepEngine

    assert FractionalStepEngine is not None


def test_fractional_step_engine_init_with_mock():
    """Test FractionalStepEngine initialization with mocked GridHandle."""
    registry = SimulationMetaRegistry()

    # Register required fields for fractional_step
    from gridfoam.CTX.application.fractional_step import FractionalStepEngine
    from gridfoam.DNA.enum import FieldLayout, FieldRole
    from gridfoam.DNA.meta.equation import EquationMeta
    from gridfoam.DNA.meta.field import FieldMeta

    U_field = FieldMeta(
        name="U",
        label="Velocity",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
        components=3,
    )
    p_field = FieldMeta(
        name="p",
        label="Pressure",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
        components=1,
    )
    phi_field = FieldMeta(
        name="phi",
        label="Flux",
        layout=FieldLayout.FACE,
        role=FieldRole.AUXILIARY,
        components=1,
    )
    registry.register_field(U_field)
    registry.register_field(p_field)
    registry.register_field(phi_field)

    momentum_eq = EquationMeta(
        name="momentum",
        target_field=U_field,
        boundary_conditions=[],
        ast_root=U_field,
    )
    registry.register_equation(momentum_eq)

    from gridfoam.DNA.config import ControlConfig, IoConfig, SolverChoice

    mock_config = MagicMock()
    mock_config.simulator.control = ControlConfig(
        writeInterval=1,
        endTime=1.0,
        deltaT=0.1,
    )
    mock_config.io = IoConfig(
        output_dir=pathlib.Path("."),
        base_name="test",
    )
    mock_config.simulator.fvSchemes = fvSchemesConfig(
        ddtSchemes={"default": FVMDdtSchemeChoice.EULER},
        divSchemes={"default": FVMDivSchemeChoice.UPWIND},
    )
    mock_config.simulator.fvSolution = fvSolutionConfig(
        solvers={
            "momentum": SolverChoice(
                method=SolverMethodChoice.CG,
                tolerance=1e-6,
                rel_tolerance=1e-6,
                max_iter=1000,
                norm_type=NormType.L_2,
            )
        }
    )

    mock_grid_handle = create_autospec(
        GridHandle, spec_set=False, instance=True
    )
    mock_grid_handle.config = mock_config

    mock_context = create_autospec(
        SimulationContext, spec_set=False, instance=True
    )
    mock_context.registry = registry
    mock_context.grid_handle = mock_grid_handle
    mock_context.registry.equations = {"momentum": momentum_eq}

    with (
        patch(
            "gridfoam.CTX.application.fractional_step.GridHandle",
            return_value=mock_grid_handle,
        ),
        patch(
            "gridfoam.CTX.context.SimulationContext", return_value=mock_context
        ),
    ):
        configpath = pathlib.Path("dummy.yaml")
        engine = FractionalStepEngine(registry=registry, configpath=configpath)

        assert engine._context is not None
        assert engine._write_interval == 1
        assert engine._end_time == 1.0
        assert engine._deltaT == 0.1
        assert engine._base_name == "test"
        assert engine._flux_correction is not None
        assert engine._rhie_chow_correction is not None
        assert hasattr(engine, "initialize")
        assert hasattr(engine, "context")


def test_fractional_step_engine_initialize_with_mock():
    """Test FractionalStepEngine initialize method with mocked dependencies."""
    registry = SimulationMetaRegistry()

    from gridfoam.CTX.application.fractional_step import FractionalStepEngine
    from gridfoam.DNA.enum import FieldLayout, FieldRole
    from gridfoam.DNA.meta.equation import EquationMeta
    from gridfoam.DNA.meta.field import FieldMeta

    U_field = FieldMeta(
        name="U",
        label="Velocity",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
        components=3,
    )
    p_field = FieldMeta(
        name="p",
        label="Pressure",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
        components=1,
    )
    phi_field = FieldMeta(
        name="phi",
        label="Flux",
        layout=FieldLayout.FACE,
        role=FieldRole.AUXILIARY,
        components=1,
    )
    registry.register_field(U_field)
    registry.register_field(p_field)
    registry.register_field(phi_field)

    momentum_eq = EquationMeta(
        name="momentum",
        target_field=U_field,
        boundary_conditions=[],
        ast_root=U_field,
    )
    registry.register_equation(momentum_eq)

    from gridfoam.DNA.config import ControlConfig, IoConfig, SolverChoice

    mock_config = MagicMock()
    mock_config.simulator.control = ControlConfig(
        writeInterval=1,
        endTime=1.0,
        deltaT=0.1,
    )
    mock_config.io = IoConfig(
        output_dir=pathlib.Path("."),
        base_name="test",
    )
    mock_config.simulator.fvSchemes = fvSchemesConfig(
        ddtSchemes={"default": FVMDdtSchemeChoice.EULER},
        divSchemes={"default": FVMDivSchemeChoice.UPWIND},
    )
    mock_config.simulator.fvSolution = fvSolutionConfig(
        solvers={
            "momentum": SolverChoice(
                method=SolverMethodChoice.CG,
                tolerance=1e-6,
                rel_tolerance=1e-6,
                max_iter=1000,
                norm_type=NormType.L_2,
            )
        }
    )

    mock_grid_handle = create_autospec(
        GridHandle, spec_set=False, instance=True
    )
    mock_grid_handle.config = mock_config
    mock_grid_handle.allocate_by_registry = MagicMock()

    mock_context = create_autospec(
        SimulationContext, spec_set=False, instance=True
    )
    mock_context.registry = registry
    mock_context.grid_handle = mock_grid_handle
    mock_context.registry.equations = {"momentum": momentum_eq}

    with (
        patch(
            "gridfoam.CTX.application.fractional_step.GridHandle",
            return_value=mock_grid_handle,
        ),
        patch(
            "gridfoam.CTX.context.SimulationContext", return_value=mock_context
        ),
    ):
        configpath = pathlib.Path("dummy.yaml")
        engine = FractionalStepEngine(registry=registry, configpath=configpath)

        engine.initialize()
        mock_grid_handle.allocate_by_registry.assert_called_once_with(registry)

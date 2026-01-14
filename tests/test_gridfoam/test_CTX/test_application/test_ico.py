"""Tests for CTX/application/ico module."""

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from gridfoam.CTX.application.ico import ICOEngine
from gridfoam.RNA.registry import SimulationMetaRegistry


def test_ico_engine_import():
    """Test that ICOEngine can be imported."""
    from gridfoam.CTX.application.ico import ICOEngine
    assert ICOEngine is not None


def test_ico_engine_init_with_mock():
    """Test ICOEngine initialization with mocked GridHandle."""
    registry = SimulationMetaRegistry()
    
    # Register required fields for ICO
    from gridfoam.DNA.enum import FieldLayout, FieldRole
    from gridfoam.DNA.meta.field import FieldMeta
    from gridfoam.DNA.meta.equation import EquationMeta
    
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
    
    mock_config = MagicMock()
    mock_config.simulator.control.writeInterval = 1
    mock_config.simulator.control.endTime = 1.0
    mock_config.simulator.control.deltaT = 0.1
    mock_config.io.base_name = "test"
    mock_config.simulator.fvSchemes = MagicMock()
    mock_config.simulator.fvSolution = MagicMock()
    
    mock_grid_handle = MagicMock()
    mock_grid_handle.config = mock_config
    
    mock_context = MagicMock()
    mock_context.registry = registry
    mock_context.grid_handle = mock_grid_handle
    mock_context.registry.equations = {"momentum": momentum_eq}
    
    with patch("gridfoam.CTX.application.ico.GridHandle", return_value=mock_grid_handle), \
         patch("gridfoam.CTX.context.SimulationContext", return_value=mock_context):
        configpath = pathlib.Path("dummy.yaml")
        engine = ICOEngine(registry=registry, configpath=configpath)
        
        assert engine._context is not None
        assert engine._write_interval == 1
        assert engine._end_time == 1.0
        assert engine._deltaT == 0.1
        assert engine._base_name == "test"
        assert engine._flux_correction is not None
        assert engine._rhie_chow_correction is not None
        assert hasattr(engine, "initialize")
        assert hasattr(engine, "context")


def test_ico_engine_initialize_with_mock():
    """Test ICOEngine initialize method with mocked dependencies."""
    registry = SimulationMetaRegistry()
    
    from gridfoam.DNA.enum import FieldLayout, FieldRole
    from gridfoam.DNA.meta.field import FieldMeta
    from gridfoam.DNA.meta.equation import EquationMeta
    
    U_field = FieldMeta(
        name="U",
        label="Velocity",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
        components=3,
    )
    registry.register_field(U_field)
    
    momentum_eq = EquationMeta(
        name="momentum",
        target_field=U_field,
        boundary_conditions=[],
        ast_root=U_field,
    )
    registry.register_equation(momentum_eq)
    
    mock_config = MagicMock()
    mock_config.simulator.control.writeInterval = 1
    mock_config.simulator.control.endTime = 1.0
    mock_config.simulator.control.deltaT = 0.1
    mock_config.io.base_name = "test"
    mock_config.simulator.fvSchemes = MagicMock()
    mock_config.simulator.fvSolution = MagicMock()
    
    mock_grid_handle = MagicMock()
    mock_grid_handle.config = mock_config
    mock_grid_handle.allocate_by_registry = MagicMock()
    
    mock_context = MagicMock()
    mock_context.registry = registry
    mock_context.grid_handle = mock_grid_handle
    mock_context.registry.equations = {"momentum": momentum_eq}
    
    with patch("gridfoam.CTX.application.ico.GridHandle", return_value=mock_grid_handle), \
         patch("gridfoam.CTX.context.SimulationContext", return_value=mock_context):
        configpath = pathlib.Path("dummy.yaml")
        engine = ICOEngine(registry=registry, configpath=configpath)
        
        engine.initialize()
        mock_grid_handle.allocate_by_registry.assert_called_once_with(registry)

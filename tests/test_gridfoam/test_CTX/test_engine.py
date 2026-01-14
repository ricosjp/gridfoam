"""Tests for CTX/engine module."""

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from gridfoam.CTX.engine import SimulationEngine
from gridfoam.RNA.registry import SimulationMetaRegistry


def test_simulation_engine_import():
    """Test that SimulationEngine can be imported."""
    from gridfoam.CTX.engine import SimulationEngine
    assert SimulationEngine is not None


def test_simulation_engine_init_with_mock():
    """Test SimulationEngine initialization with mocked GridHandle."""
    registry = SimulationMetaRegistry()
    
    # Create minimal mock config
    mock_config = MagicMock()
    mock_config.simulator.control.writeInterval = 1
    mock_config.simulator.control.endTime = 1.0
    mock_config.simulator.control.deltaT = 0.1
    mock_config.io.base_name = "test"
    mock_config.simulator.fvSchemes = MagicMock()
    mock_config.simulator.fvSolution = MagicMock()
    
    # Create minimal mock grid handle
    mock_grid_handle = MagicMock()
    mock_grid_handle.config = mock_config
    
    # Create minimal mock context
    mock_context = MagicMock()
    mock_context.registry = registry
    mock_context.grid_handle = mock_grid_handle
    
    # Patch GridHandle to return our mock
    with patch("gridfoam.CTX.engine.GridHandle", return_value=mock_grid_handle), \
         patch("gridfoam.CTX.context.SimulationContext", return_value=mock_context):
        configpath = pathlib.Path("dummy.yaml")
        engine = SimulationEngine(registry=registry, configpath=configpath)
        
        assert engine._context is not None
        assert engine._write_interval == 1
        assert engine._end_time == 1.0
        assert engine._deltaT == 0.1
        assert engine._base_name == "test"
        assert hasattr(engine, "initialize")
        assert hasattr(engine, "solve")
        assert hasattr(engine, "context")


def test_simulation_engine_initialize_with_mock():
    """Test SimulationEngine initialize method with mocked dependencies."""
    registry = SimulationMetaRegistry()
    
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
    mock_context.registry.equations = {}  # Empty equations for simplicity
    
    with patch("gridfoam.CTX.engine.GridHandle", return_value=mock_grid_handle), \
         patch("gridfoam.CTX.context.SimulationContext", return_value=mock_context):
        configpath = pathlib.Path("dummy.yaml")
        engine = SimulationEngine(registry=registry, configpath=configpath)
        
        # Initialize should call allocate_by_registry
        engine.initialize()
        
        # Verify allocate_by_registry was called
        mock_grid_handle.allocate_by_registry.assert_called_once_with(registry)


def test_simulation_engine_context_property():
    """Test SimulationEngine context property."""
    registry = SimulationMetaRegistry()
    
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
    
    with patch("gridfoam.CTX.engine.GridHandle", return_value=mock_grid_handle), \
         patch("gridfoam.CTX.context.SimulationContext", return_value=mock_context):
        configpath = pathlib.Path("dummy.yaml")
        engine = SimulationEngine(registry=registry, configpath=configpath)
        
        context = engine.context
        assert context is not None
        assert context.registry == registry
        assert context.grid_handle == mock_grid_handle

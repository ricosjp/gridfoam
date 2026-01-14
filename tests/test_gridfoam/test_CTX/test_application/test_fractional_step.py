"""Tests for CTX/application/fractional_step module."""

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from gridfoam.CTX.application.fractional_step import FractionalStepEngine
from gridfoam.RNA.registry import SimulationMetaRegistry


def test_fractional_step_engine_import():
    """Test that FractionalStepEngine can be imported."""
    from gridfoam.CTX.application.fractional_step import FractionalStepEngine
    assert FractionalStepEngine is not None


def test_fractional_step_engine_init_with_mock():
    """Test FractionalStepEngine initialization with mocked GridHandle."""
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
    mock_context.registry.equations = {}
    
    with patch("gridfoam.CTX.application.fractional_step.GridHandle", return_value=mock_grid_handle), \
         patch("gridfoam.CTX.context.SimulationContext", return_value=mock_context):
        configpath = pathlib.Path("dummy.yaml")
        engine = FractionalStepEngine(registry=registry, configpath=configpath)
        
        assert engine._context is not None
        assert engine._write_interval == 1
        assert engine._end_time == 1.0
        assert engine._deltaT == 0.1
        assert engine._base_name == "test"
        assert hasattr(engine, "initialize")
        assert hasattr(engine, "context")


def test_fractional_step_engine_context_property():
    """Test FractionalStepEngine context property."""
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
    
    with patch("gridfoam.CTX.application.fractional_step.GridHandle", return_value=mock_grid_handle), \
         patch("gridfoam.CTX.context.SimulationContext", return_value=mock_context):
        configpath = pathlib.Path("dummy.yaml")
        engine = FractionalStepEngine(registry=registry, configpath=configpath)
        
        context = engine.context
        assert context is not None
        assert context.registry == registry
        assert context.grid_handle == mock_grid_handle

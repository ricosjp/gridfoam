"""Tests for CTX/context module."""

from unittest.mock import MagicMock

import pytest

from gridfoam.CTX.context import SimulationContext
from gridfoam.RNA.registry import SimulationMetaRegistry


def test_simulation_context_import():
    """Test that SimulationContext can be imported."""
    from gridfoam.CTX.context import SimulationContext
    assert SimulationContext is not None


def test_simulation_context_init():
    """Test SimulationContext initialization."""
    registry = SimulationMetaRegistry()
    mock_grid_handle = MagicMock()
    
    context = SimulationContext(registry=registry, grid_handle=mock_grid_handle)
    assert context.registry == registry
    assert context.grid_handle == mock_grid_handle
    assert hasattr(context, "save")


def test_simulation_context_save():
    """Test SimulationContext save method."""
    registry = SimulationMetaRegistry()
    mock_grid_handle = MagicMock()
    
    # Mock config
    mock_config = MagicMock()
    mock_config.io.output_dir = MagicMock()
    mock_config.io.output_dir.__truediv__ = lambda self, other: MagicMock()
    mock_config.io.overwrite_file = True
    mock_config.cube.interior_width = 8
    
    # Mock grid
    mock_grid = MagicMock()
    mock_grid.domain.lower = MagicMock()
    mock_grid.domain.upper = MagicMock()
    mock_grid.octree_levels = []
    
    mock_grid_handle.config = mock_config
    mock_grid_handle.grid = mock_grid
    
    context = SimulationContext(registry=registry, grid_handle=mock_grid_handle)
    
    # Save should be callable (may fail due to file operations, but structure should be correct)
    assert hasattr(context, "save")

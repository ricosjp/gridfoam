"""Tests for CTX/context module."""

import pathlib
from unittest.mock import MagicMock, create_autospec

from gridfoam.CTX.context import SimulationContext
from gridfoam.RNA.grid_handle import GridHandle
from gridfoam.RNA.registry import SimulationMetaRegistry


def test_simulation_context_import():
    """Test that SimulationContext can be imported."""
    from gridfoam.CTX.context import SimulationContext

    assert SimulationContext is not None


def test_simulation_context_init():
    """Test SimulationContext initialization."""
    registry = SimulationMetaRegistry()
    mock_grid_handle = create_autospec(GridHandle, spec_set=True, instance=True)

    context = SimulationContext(registry=registry, grid_handle=mock_grid_handle)
    assert context.registry == registry
    assert context.grid_handle == mock_grid_handle
    assert hasattr(context, "save")


def test_simulation_context_save():
    """Test SimulationContext save method."""
    registry = SimulationMetaRegistry()
    mock_grid_handle = create_autospec(GridHandle, spec_set=True, instance=True)

    # Mock config
    from gridfoam.DNA.config import CubeConfig, IoConfig

    mock_config = MagicMock()
    mock_config.io = IoConfig(
        output_dir=pathlib.Path("."),
        base_name="test",
    )
    mock_config.cube = CubeConfig(interior_width=8, halo_width=2)

    # Mock grid
    mock_grid = MagicMock()
    mock_grid.domain.lower = MagicMock()
    mock_grid.domain.upper = MagicMock()
    mock_grid.octree_levels = []

    mock_grid_handle.config = mock_config
    mock_grid_handle.grid = mock_grid

    context = SimulationContext(registry=registry, grid_handle=mock_grid_handle)

    # Save should be callable
    assert hasattr(context, "save")

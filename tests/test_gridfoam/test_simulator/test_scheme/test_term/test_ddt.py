import pathlib

import pytest
import torch

from gridfoam import TensorGrid
from gridfoam._simulator._scheme._term._ddt import Ddt
from gridfoam._simulator._scheme._term._expr import Expr


@pytest.fixture
def config_path():
    """Create a config."""
    config_path = pathlib.Path("tests/data/yaml/Debug.yaml")
    return config_path


@pytest.fixture
def grid(config_path: pathlib.Path):
    """Create a test grid with cell fields."""
    grid = TensorGrid.build(config_path)
    grid.add_cell_field("U", (3,), torch.float32)
    grid.add_cell_field("T", (1,), torch.float32)
    grid.allocate_field_tensors()
    return grid


class TestDdt:
    """Test Ddt class."""

    def test_ddt_init(self) -> None:
        """Test Ddt initialization."""
        ddt = Ddt("temperature")
        assert ddt.fieldname == "temperature"

    def test_ddt_add(self) -> None:
        """Test Ddt addition with another term."""
        ddt1 = Ddt("temperature")
        ddt2 = Ddt("pressure")
        expr = ddt1 + ddt2

        assert isinstance(expr, Expr)
        assert expr.left == ddt1
        assert expr.right == ddt2
        assert expr.op == "+"

    def test_ddt_sub(self) -> None:
        """Test Ddt subtraction with another term."""
        ddt1 = Ddt("temperature")
        ddt2 = Ddt("pressure")
        expr = ddt1 - ddt2

        assert isinstance(expr, Expr)
        assert expr.left == ddt1
        assert expr.right == ddt2
        assert expr.op == "-"

    def test_ddt_matvec(self, grid: TensorGrid) -> None:
        """Test Ddt matvec method."""
        ddt = Ddt("T")
        octree_level = grid.data.octree_levels[0]
        x = torch.ones(octree_level.n_leaf_cells)
        dt = 0.001
        dx = torch.tensor([0.1, 0.1, 0.1])

        result = ddt.matvec(octree_level, x, dt, dx)

        # Check that result has correct shape
        assert result.shape == x.shape

        # Check that result is scaled by rdt
        depth = octree_level.depth
        rdt = (1 << depth) / dt
        expected = rdt * x
        torch.testing.assert_close(result, expected)

    def test_ddt_diag(self, grid: TensorGrid) -> None:
        """Test Ddt diag method."""
        ddt = Ddt("T")
        octree_level = grid.data.octree_levels[0]
        dt = 0.001
        dx = torch.tensor([0.1, 0.1, 0.1])

        result = ddt.diag(octree_level, dt, dx)

        # Check that result has correct shape
        assert result.shape == (octree_level.n_leaf_cells,)

        # Check that all elements are equal to rdt
        depth = octree_level.depth
        rdt = (1 << depth) / dt
        expected = torch.full((octree_level.n_leaf_cells,), rdt)
        torch.testing.assert_close(result, expected)

    def test_ddt_rhs(self, grid: TensorGrid) -> None:
        """Test Ddt rhs method."""
        ddt = Ddt("T")
        octree_level = grid.data.octree_levels[0]
        dt = 0.001
        dx = torch.tensor([0.1, 0.1, 0.1])

        # Fill some test data
        for cube in octree_level.nodes.values():
            if hasattr(cube, "old") and hasattr(cube.old, "cells"):
                if "T" in cube.old.cells:
                    cube.old.cells["T"].interior[:] = 1.0

        result = ddt.rhs(octree_level, dt, dx)

        # Check that result has correct shape
        assert result.shape == (octree_level.n_leaf_cells,)

        # Check that result is not all zeros (if there are leaf nodes)
        if octree_level.n_leaf_cells > 0:
            assert not torch.allclose(result, torch.zeros_like(result))

    def test_ddt_different_depths(self, grid: TensorGrid) -> None:
        """Test Ddt with different octree depths."""
        ddt = Ddt("T")
        dt = 0.001
        dx = torch.tensor([0.1, 0.1, 0.1])

        for octree_level in grid.data.octree_levels:
            depth = octree_level.depth
            rdt = (1 << depth) / dt

            # Test matvec
            x = torch.ones(octree_level.n_leaf_cells)
            result = ddt.matvec(octree_level, x, dt, dx)
            expected = rdt * x
            torch.testing.assert_close(result, expected)

            # Test diag
            diag_result = ddt.diag(octree_level, dt, dx)
            expected_diag = torch.full((octree_level.n_leaf_cells,), rdt)
            torch.testing.assert_close(diag_result, expected_diag)

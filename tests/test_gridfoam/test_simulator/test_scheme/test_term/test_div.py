import pathlib

import pytest
import torch

from gridfoam import TensorGrid
from gridfoam._simulator._scheme._interpolation._rhie_chow import (
    RhieChowInterpolation,
)
from gridfoam._simulator._scheme._term._div import Div
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


class TestDiv:
    """Test Div class."""

    def test_div_init(self) -> None:
        """Test Div initialization."""
        div = Div("U", "temperature")
        assert div.fieldname == "temperature"
        assert div.U_f_name == "_U_f"

    def test_div_add(self) -> None:
        """Test Div addition with another term."""
        div1 = Div("U", "temperature")
        div2 = Div("V", "pressure")
        expr = div1 + div2

        assert isinstance(expr, Expr)
        assert expr.left == div1
        assert expr.right == div2
        assert expr.op == "+"

    def test_div_sub(self) -> None:
        """Test Div subtraction with another term."""
        div1 = Div("U", "temperature")
        div2 = Div("V", "pressure")
        expr = div1 - div2

        assert isinstance(expr, Expr)
        assert expr.left == div1
        assert expr.right == div2
        assert expr.op == "-"

    def test_div_matvec(self, grid: TensorGrid) -> None:
        """Test Div matvec method."""
        div = Div("U", "T")
        octree_level = grid.data.octree_levels[0]
        x = torch.ones(octree_level.n_leaf_cells)
        dt = 0.001
        dx = torch.tensor([0.1, 0.1, 0.1])

        result = div.matvec(octree_level, x, dt, dx)

        # Check that result has correct shape and is zero
        assert result.shape == x.shape
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_div_diag(self, grid: TensorGrid) -> None:
        """Test Div diag method."""
        div = Div("U", "T")
        octree_level = grid.data.octree_levels[0]
        dt = 0.001
        dx = torch.tensor([0.1, 0.1, 0.1])

        result = div.diag(octree_level, dt, dx)

        # Check that result has correct shape and is zero
        assert result.shape == (octree_level.n_leaf_cells,)
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_div_rhs(self, grid: TensorGrid) -> None:
        """Test Div rhs method."""
        rhie_chow = RhieChowInterpolation("U")
        rhie_chow.run(grid)
        div = Div("U", "T")
        octree_level = grid.data.octree_levels[0]
        dt = 0.001
        dx = torch.tensor([0.1, 0.1, 0.1])

        # Fill some test data
        for cube in octree_level.nodes.values():
            if hasattr(cube, "old") and hasattr(cube.old, "cells"):
                if "T" in cube.old.cells:
                    cube.old.cells["T"].interior[:] = 1.0
            if hasattr(cube, "old") and hasattr(cube.old, "faces"):
                if "_U_f" in cube.old.faces:
                    cube.old.faces["_U_f"].x[:] = 1.0
                    cube.old.faces["_U_f"].y[:] = 1.0
                    cube.old.faces["_U_f"].z[:] = 1.0

        result = div.rhs(octree_level, dt, dx)

        # Check that result has correct shape
        assert result.shape == (octree_level.n_leaf_cells,)

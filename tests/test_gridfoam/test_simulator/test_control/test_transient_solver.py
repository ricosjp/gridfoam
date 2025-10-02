import pathlib

import pytest

from gridfoam import TensorGrid
from gridfoam._simulator._control._transient_solver import TransientSolver


@pytest.fixture
def config_path():
    """Create a config."""
    config_path = pathlib.Path("tests/data/yaml/Debug.yaml")
    return config_path


@pytest.fixture
def grid(config_path: pathlib.Path):
    """Create a test grid."""
    grid = TensorGrid.build(config_path)
    return grid


class TestTransientSolver:
    """Test TransientSolver class."""

    def test_transient_solver_init(self, grid: TensorGrid) -> None:
        """Test TransientSolver initialization."""
        solver = TransientSolver(grid)

        assert solver.grid == grid
        assert solver.deltaT == grid.config.simulator.control.deltaT
        assert solver.end_time == grid.config.simulator.control.endTime
        assert (
            solver.write_interval == grid.config.simulator.control.writeInterval
        )
        assert solver.finest_depth == grid.data.max_depth - 1
        assert solver.w_interior == grid.config.cube.interior_width
        assert solver.rhie_chow is not None
        assert solver.solver is not None

    def test_transient_solver_components(self, grid: TensorGrid) -> None:
        """Test that solver components are properly initialized."""
        solver = TransientSolver(grid)

        # Check that RhieChow interpolation is initialized
        assert solver.rhie_chow is not None
        assert hasattr(solver.rhie_chow, "U_i_name")
        assert hasattr(solver.rhie_chow, "U_f_name")
        assert solver.rhie_chow.U_i_name == "U"
        assert solver.rhie_chow.U_f_name == "_U_f"

        # Check that solver is initialized
        assert solver.solver is not None
        assert hasattr(solver.solver, "expr")
        assert hasattr(solver.solver, "max_iter")
        assert hasattr(solver.solver, "tol")

    def test_transient_solver_setup(self, grid: TensorGrid) -> None:
        """Test TransientSolver setup method."""
        solver = TransientSolver(grid)

        # Run setup
        solver.setup()

        # Check that fields were added
        assert "U" in grid.cell_field_dict
        assert "T" in grid.cell_field_dict

        # Check that tensors were allocated
        for octree_level in grid.data.octree_levels:
            for cube in octree_level.nodes.values():
                if hasattr(cube, "old") and hasattr(cube.old, "cells"):
                    assert "U" in cube.old.cells
                    assert "T" in cube.old.cells

                    # Check that velocity is set to (1,0,0)
                    u_field = cube.old.cells["U"]
                    assert u_field.interior[0].all() == 1.0
                    assert u_field.interior[1].all() == 0.0
                    assert u_field.interior[2].all() == 0.0

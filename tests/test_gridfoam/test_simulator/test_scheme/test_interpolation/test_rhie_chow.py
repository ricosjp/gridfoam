import pathlib

import pytest
import torch

from gridfoam import TensorGrid
from gridfoam._simulator._scheme._interpolation._rhie_chow import (
    RhieChowInterpolation,
)


@pytest.fixture
def config_path():
    """Create a config."""
    config_path = pathlib.Path("tests/data/yaml/Debug.yaml")
    return config_path


@pytest.fixture
def grid(config_path: pathlib.Path):
    """Create a test grid with velocity fields."""
    grid = TensorGrid.build(config_path)
    grid.add_cell_field("U", (3,), torch.float32)
    grid.allocate_field_tensors()
    return grid


class TestRhieChowInterpolation:
    """Test RhieChowInterpolation class."""

    def test_rhie_chow_init(self) -> None:
        """Test RhieChowInterpolation initialization."""
        rhie_chow = RhieChowInterpolation("U")
        assert rhie_chow.U_i_name == "U"
        assert rhie_chow.U_f_name == "_U_f"

    def test_rhie_chow_run(self, grid: TensorGrid) -> None:
        """Test RhieChowInterpolation run method."""
        rhie_chow = RhieChowInterpolation("U")

        # Fill some test data in cell-centered velocity
        for octree_level in grid.data.octree_levels:
            for cube in octree_level.nodes.values():
                if hasattr(cube, "old") and hasattr(cube.old, "cells"):
                    if "U" in cube.old.cells:
                        # Fill with test velocity data
                        cube.old.cells["U"].interior[0, :, :, :] = 1.0  # u
                        cube.old.cells["U"].interior[1, :, :, :] = 2.0  # v
                        cube.old.cells["U"].interior[2, :, :, :] = 3.0  # w

        # Run the interpolation
        rhie_chow.run(grid)

        # Check that face-centered velocities are computed
        for octree_level in grid.data.octree_levels:
            for cube in octree_level.nodes.values():
                if hasattr(cube, "old") and hasattr(cube.old, "faces"):
                    if "_U_f" in cube.old.faces:
                        face_velocity = cube.old.faces["_U_f"]
                        # Check that face velocity has correct shape
                        assert hasattr(face_velocity, "x")
                        assert hasattr(face_velocity, "y")
                        assert hasattr(face_velocity, "z")
                        assert hasattr(face_velocity, "w_interior")

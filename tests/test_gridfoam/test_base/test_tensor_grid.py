import pathlib

import pytest
import torch

from gridfoam import TensorGrid


@pytest.fixture
def config_path():
    """Create a config."""
    config_path = pathlib.Path("tests/data/yaml/DrivAer.yaml")
    return config_path


@pytest.fixture
def grid(config_path: pathlib.Path):
    """Create a test grid with cell fields."""
    grid = TensorGrid.build(config_path)
    grid.add_cell_field("T", (1,), torch.float32)
    grid.allocate_field_tensors()
    return grid


class TestTensorGrid:
    """Test TensorGrid class."""

    def test_tensor_grid_build(self, grid: TensorGrid) -> None:
        """Test TensorGrid.build method."""
        assert grid.data is not None
        assert grid.mesh is not None
        assert grid.device == torch.device("cpu")
        assert grid.cell_field_dict == {"T": ((1,), torch.float32)}
        assert grid.face_field_dict == {}

        # Verify config was loaded correctly
        assert grid.config.cube.interior_width > 0
        assert grid.config.cube.halo_width >= 0

    def test_domain_width_property(self, grid: TensorGrid) -> None:
        """Test domain_width property."""
        domain_width = grid.domain_width

        assert domain_width.shape == (3,)
        assert domain_width.dtype == torch.float32
        assert domain_width.device == torch.device("cpu")
        assert torch.all(domain_width > 0)  # Domain should have positive width

    def test_add_cell_field(self, grid: TensorGrid) -> None:
        """Test add_cell_field method."""
        # Add scalar field
        grid.add_cell_field("pressure", (1,), torch.float32)
        assert "pressure" in grid.cell_field_dict
        assert grid.cell_field_dict["pressure"] == ((1,), torch.float32)

        # Add vector field
        grid.add_cell_field("velocity", (3,), torch.float64)
        assert "velocity" in grid.cell_field_dict
        assert grid.cell_field_dict["velocity"] == ((3,), torch.float64)

        # Add tensor field
        grid.add_cell_field("stress", (2, 2), torch.int32)
        assert "stress" in grid.cell_field_dict
        assert grid.cell_field_dict["stress"] == ((2, 2), torch.int32)

    def test_add_cell_field_invalid_shape(self, grid: TensorGrid) -> None:
        """Test add_cell_field with invalid shape."""
        with pytest.raises(ValueError, match="All elements must be > 0"):
            grid.add_cell_field("invalid", (0,), torch.float32)

        with pytest.raises(ValueError, match="All elements must be > 0"):
            grid.add_cell_field("invalid", (-1, 2), torch.float32)

    def test_add_face_field(self, grid: TensorGrid) -> None:
        """Test add_face_field method."""
        # Add scalar field
        grid.add_face_field("flux", (1,), torch.float32)
        assert "flux" in grid.face_field_dict
        assert grid.face_field_dict["flux"] == ((1,), torch.float32)

        # Add vector field
        grid.add_face_field("momentum", (3,), torch.float64)
        assert "momentum" in grid.face_field_dict
        assert grid.face_field_dict["momentum"] == ((3,), torch.float64)

        # Add tensor field
        grid.add_face_field("gradient", (2, 2), torch.int32)
        assert "gradient" in grid.face_field_dict
        assert grid.face_field_dict["gradient"] == ((2, 2), torch.int32)

    def test_allocate_field_tensors(self, grid: TensorGrid) -> None:
        """Test allocate_field_tensors method."""
        grid.add_cell_field("pressure", (1,), torch.float32)
        grid.add_cell_field("velocity", (3,), torch.float32)
        grid.add_face_field("flux", (1,), torch.float32)

        grid.allocate_field_tensors()

        # Verify that octree levels have been processed
        for octree_level in grid.data.octree_levels:
            assert hasattr(octree_level, "n_cells_per_node")
            assert hasattr(octree_level, "n_cells")
            assert octree_level.n_cells_per_node > 0

            # Verify that cubes have Field instances
            for cube in octree_level.nodes.values():
                assert cube.cur is not None
                assert cube.old is not None
                assert hasattr(cube, "number")

    def test_update_halo(self, grid: TensorGrid) -> None:
        """Test update_halo method."""
        # Should not raise any exceptions
        grid.update_halo()

    def test_sync_ghost_from_parent(self, grid: TensorGrid) -> None:
        """Test sync_ghost_from_parent method."""
        # Should not raise any exceptions
        grid.sync_ghost_from_parent()

    def test_sync_ghost_from_children(self, grid: TensorGrid) -> None:
        """Test sync_ghost_from_children method."""
        # Should not raise any exceptions
        grid.sync_ghost_from_children()

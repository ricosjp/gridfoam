import pathlib
import tempfile

import pytest
import torch

from gridfoam._base.grid import Grid
from gridfoam._geometry import AABB
from gridfoam.utils.enums import Constants


@pytest.fixture
def simple_domain() -> AABB:
    """Create a simple domain for testing."""
    center = torch.tensor([1.0, 0.5, 0.5])
    halfwidth = torch.tensor([1.0, 0.5, 0.5])
    return AABB(center, halfwidth)


@pytest.fixture
def simple_grid(simple_domain: AABB) -> Grid:
    """Create a simple grid for testing."""
    level = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=torch.uint8,
    )
    global_index = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [0, 1, 0],
            [3, 3, 0],
            [4, 3, 0],
            [3, 4, 0],
            [4, 4, 0],
            [3, 3, 1],
            [4, 3, 1],
            [3, 4, 1],
            [4, 4, 1],
            [2, 1, 0],
            [3, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [2, 0, 1],
            [3, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [2, 1, 1],
            [3, 1, 1],
        ],
        dtype=torch.int32,
    )
    unit_code = 1 << (3 * (Constants.MAX_OCTREE_DEPTH - 1))
    local_code = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,
            unit_code * 0,
            unit_code * 1,
            unit_code * 2,
            unit_code * 3,
            unit_code * 4,
            unit_code * 5,
            unit_code * 6,
            unit_code * 7,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=torch.int64,
    )
    face_ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    face_ids_offset = torch.tensor(
        [0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], dtype=torch.int32
    )
    block_resolutions = torch.tensor([4, 2, 2], dtype=torch.int32)

    return Grid(
        domain=simple_domain,
        actual_max_level=2,
        block_resolutions=block_resolutions,
        level=level,
        global_index=global_index,
        local_code=local_code,
        face_ids=face_ids,
        face_ids_offset=face_ids_offset,
    )


def test_grid_initialization(simple_grid: Grid):
    """Test Grid initialization and post_init."""
    assert simple_grid.n_nodes == 23
    assert simple_grid.actual_max_level == 2
    assert torch.equal(
        simple_grid.block_resolutions,
        torch.tensor([4, 2, 2], dtype=torch.int32),
    )
    assert torch.equal(
        simple_grid.level,
        torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=torch.uint8,
        ),
    )
    assert simple_grid.field_data is not None

    # Check default fields are created
    assert hasattr(simple_grid.field_data, "U")
    assert hasattr(simple_grid.field_data, "p")
    assert simple_grid.field_data.U.shape == (23, 3)
    assert simple_grid.field_data.p.shape == (23,)


def test_add_field(simple_grid: Grid):
    """Test adding custom fields to the grid."""
    simple_grid.add_field("T", (1,), torch.float32)
    simple_grid.add_field("M", (3, 3), torch.float64)

    assert hasattr(simple_grid.field_data, "T")
    assert hasattr(simple_grid.field_data, "M")
    assert simple_grid.field_data.T.shape == (23, 1)
    assert simple_grid.field_data.M.shape == (23, 3, 3)
    assert simple_grid.field_data.T.dtype == torch.float32
    assert simple_grid.field_data.M.dtype == torch.float64


def test_get_neighbor_indices(simple_grid: Grid):
    """Test neighbor index calculation."""
    global_index = torch.tensor([[3, 3, 0]], dtype=torch.int32)

    # Test neighbor indices for level 1, global_index [3, 3, 0]
    neighbors = simple_grid.get_neighbor_indices(2, global_index)

    # Should get 26 neighbors
    assert neighbors.shape == (1, 26, 3)

    expected_neighbors = torch.tensor(
        [
            [
                [-1, -1, -1],  # [2, 2, -1]
                [2, 2, 0],
                [2, 2, 1],
                [-1, -1, -1],  # [2, 3, -1]
                [2, 3, 0],
                [2, 3, 1],
                [-1, -1, -1],  # [2, 4, -1]
                [-1, -1, -1],  # [2, 4, 0]
                [-1, -1, -1],  # [2, 4, 1]
                [-1, -1, -1],  # [3, 2, -1]
                [3, 2, 0],
                [3, 2, 1],
                [-1, -1, -1],  # [3, 3, -1]
                [3, 3, 1],
                [-1, -1, -1],  # [3, 4, -1]
                [-1, -1, -1],  # [3, 4, 0]
                [-1, -1, -1],  # [3, 4, 1]
                [-1, -1, -1],  # [4, 2, -1]
                [4, 2, 0],
                [4, 2, 1],
                [-1, -1, -1],  # [4, 3, -1]
                [4, 3, 0],
                [4, 3, 1],
                [-1, -1, -1],  # [4, 4, -1]
                [-1, -1, -1],  # [4, 4, 0]
                [-1, -1, -1],  # [4, 4, 1]
            ]
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(neighbors, expected_neighbors)


def test_calc_amr_box(simple_grid: Grid):
    """Test AMR box calculation."""
    indices = torch.tensor([[0, 0, 0], [1, 0, 1]], dtype=torch.int32)
    amr_boxes = simple_grid._calc_amr_box(indices)

    # Expected shape: (n_nodes, 6)
    # where 6 = [x_min, x_max, y_min, y_max, z_min, z_max]
    assert amr_boxes.shape == (2, 6)

    expected_amr_boxes = torch.tensor([[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1]])
    assert torch.equal(amr_boxes, expected_amr_boxes)


def test_save_structure(simple_grid: Grid):
    """Test VTK HDF file saving."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "test_grid.vtkhdf"

        # Add some test data to fields
        simple_grid.field_data.U[0] = torch.tensor([1.0, 2.0, 3.0])
        simple_grid.field_data.p[1] = 5.0

        # Save the grid structure
        simple_grid.save_structure(file_path)

        # Verify file was created
        assert file_path.exists()

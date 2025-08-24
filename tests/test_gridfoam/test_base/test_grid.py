import pathlib
import tempfile

import pytest
import torch

from gridfoam._base._cube import Cube
from gridfoam._base.grid import Grid
from gridfoam._geometry import AABB
from gridfoam.settings import CubeSetting, FieldDataAttribute
from gridfoam.utils.enums import CubeType, GridCalculationMode


@pytest.fixture
def basic_aabb() -> AABB:
    """Create a basic AABB for testing."""
    return AABB(
        min_pt=torch.tensor([0.0, 0.0, 0.0]),
        max_pt=torch.tensor([2.0, 1.0, 1.0]),
    )


@pytest.fixture
def basic_cube_setting() -> CubeSetting:
    """Create a basic CubeSetting for testing."""
    return CubeSetting(width=8, bnd_width=2)

@pytest.fixture
def basic_field_data_dict() -> dict[str, FieldDataAttribute]:
    """Create a basic FieldDataAttribute for testing."""
    return {
        "U": FieldDataAttribute(shape=(3,), dtype=torch.float32),
        "p": FieldDataAttribute(shape=(1,), dtype=torch.float32),
    }

@pytest.fixture
def basic_cubes() -> dict[int, dict[int, Cube]]:
    """Create basic cubes for testing."""
    cubes = {}
    depth0_global_indices = torch.tensor(
        [
            [0, 0, 0],
            [3, 0, 0],
            [0, 1, 0],
            [3, 1, 0],
            [0, 0, 1],
            [3, 0, 1],
            [0, 1, 1],
            [3, 1, 1],
        ],
        dtype=torch.int32,
    )
    depth1_global_indices = torch.tensor(
        [
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],
            [5, 0, 0],
            [2, 1, 0],
            [3, 1, 0],
            [4, 1, 0],
            [5, 1, 0],
            [2, 2, 0],
            [3, 2, 0],
            [4, 2, 0],
            [5, 2, 0],
            [2, 3, 0],
            [3, 3, 0],
            [4, 3, 0],
            [5, 3, 0],
            [2, 0, 1],
            [3, 0, 1],
            [4, 0, 1],
            [5, 0, 1],
            [2, 1, 1],
            [3, 1, 1],
            [4, 1, 1],
            [5, 1, 1],
            [2, 2, 1],
            [3, 2, 1],
            [4, 2, 1],
            [5, 2, 1],
            [2, 3, 1],
            [3, 3, 1],
            [4, 3, 1],
            [5, 3, 1],
            [2, 0, 2],
            [3, 0, 2],
            [4, 0, 2],
            [5, 0, 2],
            [2, 1, 2],
            [3, 1, 2],
            [4, 1, 2],
            [5, 1, 2],
            [2, 2, 2],
            [3, 2, 2],
            [4, 2, 2],
            [5, 2, 2],
            [2, 3, 2],
            [3, 3, 2],
            [4, 3, 2],
            [5, 3, 2],
            [2, 0, 3],
            [3, 0, 3],
            [4, 0, 3],
            [5, 0, 3],
            [2, 1, 3],
            [3, 1, 3],
            [4, 1, 3],
            [5, 1, 3],
            [2, 2, 3],
            [3, 2, 3],
            [4, 2, 3],
            [5, 2, 3],
            [2, 3, 3],
            [3, 3, 3],
            [4, 3, 3],
            [5, 3, 3],
        ],
        dtype=torch.int32,
    )

    # Create cubes for depth 0
    cubes[0] = {}
    for i, global_index in enumerate(depth0_global_indices):
        cube = Cube(
            cube_type=CubeType.LEAF,
            depth=0,
            global_index=global_index,
            face_ids=torch.tensor([], dtype=torch.int32),
            device=torch.device("cpu"),
        )
        cubes[0][i] = cube

    # Create cubes for depth 1
    cubes[1] = {}
    for i, global_index in enumerate(depth1_global_indices):
        cube = Cube(
            cube_type=CubeType.LEAF,
            depth=1,
            global_index=global_index,
            face_ids=torch.tensor([], dtype=torch.int32),
            device=torch.device("cpu"),
        )
        cubes[1][i] = cube

    return cubes


class TestGridInitialization:
    """Test Grid initialization."""

    def test_init_basic(
        self,
        basic_aabb: AABB,
        basic_cube_setting: CubeSetting,
        basic_field_data_dict: dict[str, FieldDataAttribute],
        basic_cubes: dict[int, dict[int, Cube]],
    ):
        """Test basic Grid initialization."""
        block_divisions = torch.tensor([4, 2, 2], dtype=torch.int32)
        actual_max_depth = 1

        grid = Grid(
            domain=basic_aabb,
            actual_max_depth=actual_max_depth,
            block_divisions=block_divisions,
            cubes=basic_cubes,
            cube_setting=basic_cube_setting,
            field_data_dict=basic_field_data_dict,
        )

        assert grid.domain == basic_aabb
        assert grid.actual_max_depth == actual_max_depth
        assert torch.equal(grid.block_divisions, block_divisions)
        assert grid.cubes == basic_cubes
        assert grid.cube_setting == basic_cube_setting
        assert grid.device == torch.device("cpu")


@pytest.fixture
def basic_grid(
    basic_aabb: AABB,
    basic_cube_setting: CubeSetting,
    basic_field_data_dict: dict[str, FieldDataAttribute],
    basic_cubes: dict[int, dict[int, Cube]],
) -> Grid:
    """Create a basic Grid for testing."""
    return Grid(
        domain=basic_aabb,
        actual_max_depth=1,
        block_divisions=torch.tensor([4, 2, 2], dtype=torch.int32),
        cubes=basic_cubes,
        cube_setting=basic_cube_setting,
        field_data_dict=basic_field_data_dict,
    )


class TestGridCalcAmrBox:
    """Test Grid _calc_amr_box method."""

    def test_calc_amr_box_node_basic(self, basic_grid: Grid):
        """Test basic _calc_amr_box functionality."""
        # Test with single index
        indices = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        result = basic_grid._calc_amr_box(indices, mode=GridCalculationMode.NODE)

        # Check that result contains [min, max] for each dimension
        expected = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_calc_amr_box_node_multiple_indices(self, basic_grid: Grid):
        """Test _calc_amr_box with multiple indices."""

        # Test with multiple indices
        indices = torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.int32)
        result = basic_grid._calc_amr_box(indices, mode=GridCalculationMode.NODE)

        # Check that result contains [min, max] for each dimension
        expected = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(result, expected)

    def test_calc_amr_box_cell_basic(self, basic_grid: Grid):
        """Test basic _calc_amr_box functionality."""
        # Test with single index
        indices = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        result = basic_grid._calc_amr_box(indices, mode=GridCalculationMode.CELL)

        # Check that result contains [min, max] for each dimension
        expected = torch.tensor(
            [[0, 7, 0, 7, 0, 7]],
            dtype=torch.int32,
        )
        torch.testing.assert_close(result, expected)

    def test_calc_amr_box_cell_multiple_indices(self, basic_grid: Grid):
        """Test _calc_amr_box with multiple indices."""
        # Test with multiple indices
        indices = torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.int32)
        result = basic_grid._calc_amr_box(indices, mode=GridCalculationMode.CELL)

        # Check that result contains [min, max] for each dimension
        expected = torch.tensor(
            [[0, 7, 0, 7, 0, 7], [8, 15, 8, 15, 0, 7]],
            dtype=torch.int32,
        )
        torch.testing.assert_close(result, expected)


class TestGridSaveStructure:
    """Test Grid save_structure method."""

    def test_save_structure_basic(self, basic_grid: Grid):
        """Test basic save_structure functionality."""

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = pathlib.Path(temp_dir) / "test_grid.vtkhdf"

            # Should not raise any exceptions
            basic_grid.save_structure(file_path)

            # Check that file was created
            assert file_path.exists()

    def test_save_grid_basic(self, basic_grid: Grid):
        """Test basic save_grid functionality."""

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = pathlib.Path(temp_dir) / "test_grid.vtkhdf"

            # Should not raise any exceptions
            basic_grid.save_grid(file_path)

            # Check that file was created
            assert file_path.exists()

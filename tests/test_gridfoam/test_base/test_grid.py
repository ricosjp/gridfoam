import pathlib
import tempfile

import pytest
import torch

from gridfoam._base._cube import Cube
from gridfoam._base.grid import Grid
from gridfoam._geometry import AABB
from gridfoam.settings import CubeSetting, FieldDataAttribute
from gridfoam.utils.cube_code import global_indices_to_codes
from gridfoam.utils.enums import (
    CubeType,
    GridCalculationMode,
)


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
        result = basic_grid._calc_amr_box(
            indices, mode=GridCalculationMode.NODE
        )

        # Check that result contains [min, max] for each dimension
        expected = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_calc_amr_box_node_multiple_indices(self, basic_grid: Grid):
        """Test _calc_amr_box with multiple indices."""

        # Test with multiple indices
        indices = torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.int32)
        result = basic_grid._calc_amr_box(
            indices, mode=GridCalculationMode.NODE
        )

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
        result = basic_grid._calc_amr_box(
            indices, mode=GridCalculationMode.CELL
        )

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
        result = basic_grid._calc_amr_box(
            indices, mode=GridCalculationMode.CELL
        )

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


@pytest.fixture
def grid_with_field_tensors(basic_grid: Grid) -> Grid:
    """Create a Grid with allocated field tensors for testing."""
    basic_grid.allocate_field_tensors()
    return basic_grid


class TestGridAllocateFieldTensors:
    """Test Grid allocate_field_tensors method."""

    def test_allocate_field_tensors_basic(self, basic_grid: Grid):
        """Test basic allocate_field_tensors functionality."""
        # Before allocation, field_tensors should be empty
        for cubes_by_depth in basic_grid.cubes.values():
            for cube in cubes_by_depth.values():
                assert (
                    not hasattr(cube, "field_tensors")
                    or len(cube.field_tensors) == 0
                )

        # Allocate field tensors
        basic_grid.allocate_field_tensors()

        # After allocation, field_tensors should contain expected fields
        for cubes_by_depth in basic_grid.cubes.values():
            for cube in cubes_by_depth.values():
                assert hasattr(cube, "field_tensors")
                assert "U" in cube.field_tensors
                assert "p" in cube.field_tensors

    def test_allocate_field_tensors_field_shapes(self, basic_grid: Grid):
        """Test that allocated field tensors have correct shapes."""
        basic_grid.allocate_field_tensors()

        for cubes_by_depth in basic_grid.cubes.values():
            for cube in cubes_by_depth.values():
                # Check U field shape (3,)
                assert cube.field_tensors["U"].interior.shape[-1] == 3
                # Check p field shape (1,)
                assert cube.field_tensors["p"].interior.shape[-1] == 1


class TestGridSpacing:
    """Test Grid spacing method."""

    def test_spacing_cell_mode_depth_0(self, basic_grid: Grid):
        """Test spacing calculation for depth 0 in CELL mode."""
        spacing = basic_grid.spacing(0, GridCalculationMode.CELL)

        # Expected spacing for depth 0:
        #   domain_width / (block_divisions * 2^0 * width)
        # domain_width = [2.0, 1.0, 1.0]
        # block_divisions = [4, 2, 2]
        # width = 8
        # Expected: [2.0, 1.0, 1.0] / (4 * 1 * 8) = [0.0625, 0.0625, 0.0625]
        expected = torch.tensor([0.0625, 0.0625, 0.0625], dtype=torch.float32)
        torch.testing.assert_close(spacing, expected)

    def test_spacing_cell_mode_depth_1(self, basic_grid: Grid):
        """Test spacing calculation for depth 1 in CELL mode."""
        spacing = basic_grid.spacing(1, GridCalculationMode.CELL)

        # Expected spacing for depth 1:
        #   domain_width / (block_divisions * 2^1 * width)
        # Expected: [2.0, 1.0, 1.0] / (4 * 2 * 8) = [0.03125, 0.03125, 0.03125]
        expected = torch.tensor(
            [0.03125, 0.03125, 0.03125], dtype=torch.float32
        )
        torch.testing.assert_close(spacing, expected)

    def test_spacing_node_mode_depth_0(self, basic_grid: Grid):
        """Test spacing calculation for depth 0 in NODE mode."""
        spacing = basic_grid.spacing(0, GridCalculationMode.NODE)

        # Expected spacing for depth 0: domain_width / (block_divisions * 2^0)
        # Expected: [2.0, 1.0, 1.0] / (4 * 1) = [0.5, 0.5, 0.5]
        expected = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        torch.testing.assert_close(spacing, expected)

    def test_spacing_node_mode_depth_1(self, basic_grid: Grid):
        """Test spacing calculation for depth 1 in NODE mode."""
        spacing = basic_grid.spacing(1, GridCalculationMode.NODE)

        # Expected spacing for depth 1: domain_width / (block_divisions * 2^1)
        # Expected: [2.0, 1.0, 1.0] / (4 * 2) = [0.25, 0.25, 0.25]
        expected = torch.tensor([0.25, 0.25, 0.25], dtype=torch.float32)
        torch.testing.assert_close(spacing, expected)


@pytest.fixture
def grid_with_neighbors(
    basic_aabb: AABB,
    basic_cube_setting: CubeSetting,
    basic_field_data_dict: dict[str, FieldDataAttribute],
) -> Grid:
    """Create a Grid with neighboring cubes for testing update_halo."""
    # Create a simple 2x2x2 grid at depth 0 for testing
    cubes = {}
    depth0_global_indices = torch.tensor(
        [
            [0, 0, 0],  # Cube 0
            [1, 0, 0],  # Cube 1 (neighbor of 0 in +x)
            [0, 1, 0],  # Cube 2 (neighbor of 0 in +y)
            [1, 1, 0],  # Cube 3
        ],
        dtype=torch.int32,
    )

    # Create cubes for depth 0
    cubes[0] = {}
    codes = global_indices_to_codes(
        depth0_global_indices, torch.tensor([2, 2, 1], dtype=torch.int32), 0
    )
    for code, global_index in zip(codes, depth0_global_indices, strict=True):
        cube = Cube(
            cube_type=CubeType.LEAF,
            depth=0,
            global_index=global_index,
            face_ids=torch.tensor([], dtype=torch.int32),
            device=torch.device("cpu"),
        )
        cubes[0][code] = cube

    grid = Grid(
        domain=basic_aabb,
        actual_max_depth=0,
        block_divisions=torch.tensor([2, 2, 1], dtype=torch.int32),
        cubes=cubes,
        cube_setting=basic_cube_setting,
        field_data_dict=basic_field_data_dict,
    )

    # Allocate field tensors
    grid.allocate_field_tensors()

    # Initialize some test data
    for cube in cubes[0].values():
        cube.field_tensors["p"].interior = (
            torch.ones_like(cube.field_tensors["p"].interior) * 1.0
        )
        cube.field_tensors["U"].interior = (
            torch.ones_like(cube.field_tensors["U"].interior) * 2.0
        )

    return grid


class TestGridUpdateHalo:
    """Test Grid update_halo method."""

    def test_update_halo_basic(self, grid_with_neighbors: Grid):
        """Test basic update_halo functionality."""
        # Get a cube to test
        cube = grid_with_neighbors.cubes[0][0]  # Cube at [0, 0, 0]

        # Before update_halo, halo regions are all zeros
        assert torch.all(cube.field_tensors["p"].xp == 0.0)
        assert torch.all(cube.field_tensors["p"].yp == 0.0)

        grid_with_neighbors.update_halo()

        # After update_halo, halo regions are updated
        assert torch.all(cube.field_tensors["p"].xp == 1.0)
        assert torch.all(cube.field_tensors["p"].yp == 1.0)


@pytest.fixture
def grid_with_ghost(
    basic_aabb: AABB,
    basic_cube_setting: CubeSetting,
    basic_field_data_dict: dict[str, FieldDataAttribute],
) -> Grid:
    """Create a Grid with ghost cubes for testing ghost sync."""
    cubes = {}

    # Create parent cube at depth 0
    parent_cube = Cube(
        cube_type=CubeType.LEAF,
        depth=0,
        global_index=torch.tensor([0, 0, 0], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    # Create ghost cube that will get data from children
    ghost_from_children_1 = Cube(
        cube_type=CubeType.GHOST_FROM_CHILD,
        depth=0,
        global_index=torch.tensor([1, 0, 0], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    depth0_codes = global_indices_to_codes(
        torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32),
        torch.tensor([2, 1, 1], dtype=torch.int32),
        0,
    )
    cubes[0] = {
        depth0_codes[0]: parent_cube,
        depth0_codes[1]: ghost_from_children_1,
    }

    # Create child cubes at depth 1
    # Create ghost cube that will get data from parent
    ghost_from_parent_0 = Cube(
        cube_type=CubeType.GHOST_FROM_PARENT,
        depth=1,
        global_index=torch.tensor([0, 0, 0], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    ghost_from_parent_1 = Cube(
        cube_type=CubeType.GHOST_FROM_PARENT,
        depth=1,
        global_index=torch.tensor([1, 0, 0], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    ghost_from_parent_2 = Cube(
        cube_type=CubeType.GHOST_FROM_PARENT,
        depth=1,
        global_index=torch.tensor([0, 1, 0], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    ghost_from_parent_3 = Cube(
        cube_type=CubeType.GHOST_FROM_PARENT,
        depth=1,
        global_index=torch.tensor([1, 1, 0], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    ghost_from_parent_4 = Cube(
        cube_type=CubeType.GHOST_FROM_PARENT,
        depth=1,
        global_index=torch.tensor([0, 0, 1], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    ghost_from_parent_5 = Cube(
        cube_type=CubeType.GHOST_FROM_PARENT,
        depth=1,
        global_index=torch.tensor([1, 0, 1], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    ghost_from_parent_6 = Cube(
        cube_type=CubeType.GHOST_FROM_PARENT,
        depth=1,
        global_index=torch.tensor([0, 1, 1], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    ghost_from_parent_7 = Cube(
        cube_type=CubeType.GHOST_FROM_PARENT,
        depth=1,
        global_index=torch.tensor([1, 1, 1], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    child_cube0 = Cube(
        cube_type=CubeType.LEAF,
        depth=1,
        global_index=torch.tensor([2, 0, 0], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    child_cube1 = Cube(
        cube_type=CubeType.LEAF,
        depth=1,
        global_index=torch.tensor([3, 0, 0], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    child_cube2 = Cube(
        cube_type=CubeType.LEAF,
        depth=1,
        global_index=torch.tensor([2, 1, 0], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    child_cube3 = Cube(
        cube_type=CubeType.LEAF,
        depth=1,
        global_index=torch.tensor([3, 1, 0], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    child_cube4 = Cube(
        cube_type=CubeType.LEAF,
        depth=1,
        global_index=torch.tensor([2, 0, 1], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    child_cube5 = Cube(
        cube_type=CubeType.LEAF,
        depth=1,
        global_index=torch.tensor([3, 0, 1], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    child_cube6 = Cube(
        cube_type=CubeType.LEAF,
        depth=1,
        global_index=torch.tensor([2, 1, 1], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    child_cube7 = Cube(
        cube_type=CubeType.LEAF,
        depth=1,
        global_index=torch.tensor([3, 1, 1], dtype=torch.int32),
        face_ids=torch.tensor([], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    depth1_codes = global_indices_to_codes(
        torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [2, 0, 0],
                [3, 0, 0],
                [2, 1, 0],
                [3, 1, 0],
                [2, 0, 1],
                [3, 0, 1],
                [2, 1, 1],
                [3, 1, 1],
            ],
            dtype=torch.int32,
        ),
        torch.tensor([2, 1, 1], dtype=torch.int32),
        1,
    )
    cubes[1] = {
        depth1_codes[0]: ghost_from_parent_0,
        depth1_codes[1]: ghost_from_parent_1,
        depth1_codes[2]: ghost_from_parent_2,
        depth1_codes[3]: ghost_from_parent_3,
        depth1_codes[4]: ghost_from_parent_4,
        depth1_codes[5]: ghost_from_parent_5,
        depth1_codes[6]: ghost_from_parent_6,
        depth1_codes[7]: ghost_from_parent_7,
        depth1_codes[8]: child_cube0,
        depth1_codes[9]: child_cube1,
        depth1_codes[10]: child_cube2,
        depth1_codes[11]: child_cube3,
        depth1_codes[12]: child_cube4,
        depth1_codes[13]: child_cube5,
        depth1_codes[14]: child_cube6,
        depth1_codes[15]: child_cube7,
    }

    grid = Grid(
        domain=basic_aabb,
        actual_max_depth=1,
        block_divisions=torch.tensor([2, 1, 1], dtype=torch.int32),
        cubes=cubes,
        cube_setting=basic_cube_setting,
        field_data_dict=basic_field_data_dict,
    )

    # Allocate field tensors
    grid.allocate_field_tensors()
    return grid


class TestGridSyncGhostFromParent:
    """Test Grid sync_ghost_from_parent method."""

    def test_sync_ghost_from_parent_basic(self, grid_with_ghost: Grid):
        """Test basic sync_ghost_from_parent functionality."""
        dtype = torch.float32
        for depth, cube in grid_with_ghost.iterate_by_depth():
            cube.field_tensors["p"].interior = torch.tensor(
                [depth + 1], dtype=dtype
            )

        grid_with_ghost.sync_ghost_from_parent()

        expected = torch.ones((8, 8, 8, 1), dtype=dtype)
        for cube in grid_with_ghost.cubes[1].values():
            if cube.cube_type == CubeType.GHOST_FROM_PARENT:
                assert torch.allclose(
                    cube.field_tensors["p"].interior, expected
                )


class TestGridSyncGhostFromChildren:
    """Test Grid sync_ghost_from_children method."""

    def test_sync_ghost_from_children_basic(self, grid_with_ghost: Grid):
        """Test basic sync_ghost_from_children functionality."""
        dtype = torch.float32
        for depth, cube in grid_with_ghost.iterate_by_depth():
            cube.field_tensors["p"].interior = torch.tensor(
                [depth], dtype=dtype
            )

        grid_with_ghost.sync_ghost_from_children()

        expected = torch.ones((8, 8, 8, 1), dtype=dtype)
        for cube in grid_with_ghost.cubes[0].values():
            if cube.cube_type == CubeType.GHOST_FROM_CHILD:
                assert torch.allclose(
                    cube.field_tensors["p"].interior, expected
                )

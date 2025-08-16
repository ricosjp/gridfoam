import pytest
import torch
from trimesh.triangles import bounds_tree

from gridfoam._base.grid import Grid
from gridfoam._geometry import AABB, TriangleMesh
from gridfoam._octree._forest import Forest
from gridfoam._octree._node import OctreeNode
from gridfoam.settings import GridSetting
from gridfoam.utils.cube_code import global_indices_to_codes
from gridfoam.utils.enums import CubeType


@pytest.fixture
def basic_grid_setting() -> GridSetting:
    """Create a basic GridSetting for testing."""
    return GridSetting(
        blockXMin=-2.0,
        blockXMax=2.0,
        blockYMin=-1.0,
        blockYMax=1.0,
        blockZMin=-1.0,
        blockZMax=1.0,
        nBlockX=4,
        nBlockY=2,
        nBlockZ=2,
        alpha=0.3,
        depth_limit=3,
    )


@pytest.fixture
def simple_mesh() -> TriangleMesh:
    """Create a simple mesh for testing."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [4, 7, 5],
            [5, 7, 6],
        ]
    )
    gaussian_curvatures = torch.tensor([0.1] * 8)
    tree = bounds_tree(points[faces])
    return TriangleMesh(
        points=points,
        faces=faces,
        gaussian_curvatures=gaussian_curvatures,
        tree=tree,
    )


@pytest.fixture
def high_curvature_mesh() -> TriangleMesh:
    """Create a mesh with high curvature for testing splitting."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [4, 7, 5],
            [5, 7, 6],
        ]
    )
    # High curvature values to trigger splitting
    gaussian_curvatures = torch.ones(8) * 10.0
    tree = bounds_tree(points[faces])
    return TriangleMesh(
        points=points,
        faces=faces,
        gaussian_curvatures=gaussian_curvatures,
        tree=tree,
    )


class TestForestInitialization:
    """Test Forest initialization."""

    def test_init_basic(self, basic_grid_setting: GridSetting):
        """Test basic Forest initialization."""
        forest = Forest(basic_grid_setting)

        assert torch.equal(forest._bbox.min, torch.tensor([-2.0, -1.0, -1.0]))
        assert torch.equal(forest._bbox.max, torch.tensor([2.0, 1.0, 1.0]))
        assert torch.equal(
            forest._block_divisions, torch.tensor([4, 2, 2], dtype=torch.int32)
        )
        assert forest._alpha == 0.3
        assert forest._depth_limit == 3
        assert forest._actual_max_depth == 0
        assert forest._nodes == {}
        assert forest._cubes == {}
        assert forest._device == torch.device("cpu")

    def test_init_custom_device(self):
        """Test Forest initialization with custom device."""
        grid_setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=2,
            nBlockY=2,
            device="cuda:0",
        )
        forest = Forest(grid_setting)
        assert forest._device == torch.device("cuda:0")

    def test_init_custom_alpha_and_depth(self):
        """Test Forest initialization with custom alpha and depth_limit."""
        grid_setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=2,
            nBlockY=2,
            alpha=0.5,
            depth_limit=5,
        )
        forest = Forest(grid_setting)
        assert forest._alpha == 0.5
        assert forest._depth_limit == 5


class TestForestShouldSplitNode:
    """Test Forest _should_split_node method."""

    def test_should_split_node_curvature_condition(
        self, basic_grid_setting: GridSetting, high_curvature_mesh: TriangleMesh
    ):
        """Test _should_split_node with high curvature."""
        forest = Forest(basic_grid_setting)

        # Create a node that should be split due to high curvature
        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([0.5, 0.5, 0.5]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=0,
            morton_code=0,
            face_ids=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

        # Should split due to high curvature and depth < limit
        assert forest._should_split_node(node, high_curvature_mesh) is True

    def test_should_split_node_depth_limit(
        self, basic_grid_setting: GridSetting, high_curvature_mesh: TriangleMesh
    ):
        """Test _should_split_node with depth limit."""
        forest = Forest(basic_grid_setting)

        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([0.5, 0.5, 0.5]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=3,  # At depth limit
            morton_code=0,
            face_ids=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

        # Should not split due to depth limit
        assert forest._should_split_node(node, high_curvature_mesh) is False

    def test_should_split_node_low_curvature(
        self, basic_grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test _should_split_node with low curvature."""
        forest = Forest(basic_grid_setting)

        from gridfoam._octree._node import OctreeNode

        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([0.5, 0.5, 0.5]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=0,
            morton_code=0,
            face_ids=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

        # Should not split due to low curvature
        assert forest._should_split_node(node, simple_mesh) is False


class TestForestDilateCoords:
    """Test Forest _dilate_coords method."""

    def test_dilate_coords_basic(self, basic_grid_setting: GridSetting):
        """Test basic _dilate_coords functionality."""
        forest = Forest(basic_grid_setting)
        block_divisions = forest._block_divisions

        coords_list = [[0, 0, 0]]
        octree_depth = 0

        result = forest._dilate_coords(coords_list, octree_depth)
        expected_coords = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.int32,
        )
        expected_codes = global_indices_to_codes(
            expected_coords, block_divisions, octree_depth
        )
        torch.testing.assert_close(result, expected_codes)

    def test_dilate_coords_multiple_coords(
        self, basic_grid_setting: GridSetting
    ):
        """Test _dilate_coords with multiple coordinates."""
        forest = Forest(basic_grid_setting)
        block_divisions = forest._block_divisions

        coords_list = [[0, 0, 0], [1, 1, 1]]
        octree_depth = 0

        result = forest._dilate_coords(coords_list, octree_depth)
        expected_coords = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [2, 0, 0],
                [2, 0, 1],
                [2, 1, 0],
                [2, 1, 1],
            ],
            dtype=torch.int32,
        )
        expected_codes = global_indices_to_codes(
            expected_coords, block_divisions, octree_depth
        )
        torch.testing.assert_close(result, expected_codes)

    def test_dilate_coords_deeper_depth(self, basic_grid_setting: GridSetting):
        """Test _dilate_coords with deeper octree depth."""
        forest = Forest(basic_grid_setting)
        block_divisions = forest._block_divisions

        coords_list = [[1, 1, 1]]
        octree_depth = 1

        result = forest._dilate_coords(coords_list, octree_depth)
        expected_coords = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 0],
                [0, 2, 1],
                [0, 2, 2],
                [1, 0, 0],
                [1, 0, 1],
                [1, 0, 2],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 2, 0],
                [1, 2, 1],
                [1, 2, 2],
                [2, 0, 0],
                [2, 0, 1],
                [2, 0, 2],
                [2, 1, 0],
                [2, 1, 1],
                [2, 1, 2],
                [2, 2, 0],
                [2, 2, 1],
                [2, 2, 2],
            ],
            dtype=torch.int32,
        )
        expected_codes = global_indices_to_codes(
            expected_coords, block_divisions, octree_depth
        )
        torch.testing.assert_close(result, expected_codes)


class TestForestGenerateRoots:
    """Test Forest _generate_roots method."""

    def test_generate_roots_basic(
        self, basic_grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test basic _generate_roots functionality."""
        forest = Forest(basic_grid_setting)
        forest._generate_roots(simple_mesh)

        # Should create 8 root nodes (4x2x2)
        assert len(forest._nodes[0]) == 16

        # Check that all root nodes have correct properties
        for node in forest._nodes[0].values():
            assert node.octree_depth == 0
            assert node.morton_code == 0
            assert node.is_leaf

    def test_generate_roots_single_block(self, simple_mesh: TriangleMesh):
        """Test _generate_roots with single block division."""
        grid_setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=1,
            nBlockY=1,
            nBlockZ=1,
        )
        forest = Forest(grid_setting)
        forest._generate_roots(simple_mesh)

        # Should create 1 root node
        assert len(forest._nodes[0]) == 1

        root_node = list(forest._nodes[0].values())[0]
        assert root_node.octree_depth == 0
        assert root_node.morton_code == 0

    def test_generate_roots_face_ids(
        self, basic_grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test that root nodes have correct face_ids."""
        forest = Forest(basic_grid_setting)
        forest._generate_roots(simple_mesh)

        # Check that face_ids are properly assigned
        for node in forest._nodes[0].values():
            assert isinstance(node.face_ids, torch.Tensor)
            # All face_ids should be valid indices
            for face_id in node.face_ids:
                assert 0 <= face_id < simple_mesh.n_triangles


class TestForestRecursiveSplitByMesh:
    """Test Forest _recursive_split_by_mesh method."""

    def test_recursive_split_by_mesh_basic(
        self, basic_grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test basic _recursive_split_by_mesh functionality."""
        forest = Forest(basic_grid_setting)
        forest._generate_roots(simple_mesh)

        forest._recursive_split_by_mesh(simple_mesh)

        # depth should be 0
        assert len(forest._nodes) == 1
        assert forest._actual_max_depth == 0

    def test_recursive_split_by_mesh_with_high_curvature(
        self, basic_grid_setting: GridSetting, high_curvature_mesh: TriangleMesh
    ):
        """Test _recursive_split_by_mesh with high curvature mesh."""
        forest = Forest(basic_grid_setting)
        forest._generate_roots(high_curvature_mesh)

        forest._recursive_split_by_mesh(high_curvature_mesh)

        # Should have more nodes due to splitting
        assert len(forest._nodes) > 1

        # Should reach deeper levels due to high curvature
        assert forest._actual_max_depth > 0

    def test_recursive_split_by_mesh_depth_limit(
        self, high_curvature_mesh: TriangleMesh
    ):
        """Test _recursive_split_by_mesh respects depth limit."""
        grid_setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=2,
            nBlockY=2,
            depth_limit=1,  # Very low depth limit
        )
        forest = Forest(grid_setting)
        forest._generate_roots(high_curvature_mesh)

        forest._recursive_split_by_mesh(high_curvature_mesh)

        # Should not exceed depth limit
        assert len(forest._nodes) <= 2
        assert forest._actual_max_depth <= 1


class TestForestGenerateCubes:
    """Test Forest _generate_cubes method."""

    def test_generate_cubes_basic(
        self, basic_grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test basic _generate_cubes functionality."""
        forest = Forest(basic_grid_setting)

        # First generate roots and split some nodes
        forest._generate_roots(simple_mesh)
        forest._recursive_split_by_mesh(simple_mesh)

        # Then generate cubes
        forest._generate_cubes()

        # Should have cubes for each depth
        for depth in forest._nodes:
            assert depth in forest._cubes
            assert len(forest._cubes[depth]) > 0

    def test_generate_cubes_leaf_and_ghost(
        self, basic_grid_setting: GridSetting, high_curvature_mesh: TriangleMesh
    ):
        """Test that both leaf and ghost cubes are generated."""
        forest = Forest(basic_grid_setting)

        # Generate a simple structure
        forest._generate_roots(high_curvature_mesh)
        forest._recursive_split_by_mesh(high_curvature_mesh)
        forest._generate_cubes()

        # Check that we have both leaf and ghost cubes
        leaf_cubes = 0
        ghost_cubes = 0

        for _, cubes in forest._cubes.items():
            for cube in cubes.values():
                if cube.cube_type == CubeType.LEAF:
                    leaf_cubes += 1
                elif cube.cube_type in [
                    CubeType.GHOST_FROM_PARENT,
                    CubeType.GHOST_FROM_CHILD,
                ]:
                    ghost_cubes += 1

        assert leaf_cubes > 0
        assert ghost_cubes > 0


class TestForestBuildGridFromMesh:
    """Test Forest build_grid_from_mesh method."""

    def test_build_grid_from_mesh_basic(
        self, basic_grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test basic build_grid_from_mesh functionality."""
        forest = Forest(basic_grid_setting)

        grid = forest.build_grid_from_mesh(simple_mesh)

        # Should return a Grid object
        assert isinstance(grid, Grid)

        # Should have correct properties
        assert grid.domain == forest._bbox
        assert grid.actual_max_depth == forest._actual_max_depth
        assert torch.equal(grid.block_divisions, forest._block_divisions)
        assert grid.cube_setting == forest._cube_setting
        assert grid.device == forest._device

    def test_build_grid_from_mesh_with_splitting(
        self, basic_grid_setting: GridSetting, high_curvature_mesh: TriangleMesh
    ):
        """Test build_grid_from_mesh with mesh that causes splitting."""
        forest = Forest(basic_grid_setting)

        grid = forest.build_grid_from_mesh(high_curvature_mesh)

        # Should have cubes at multiple depths
        assert len(grid.cubes) > 1

        # Should have actual_max_depth > 0
        assert grid.actual_max_depth > 0

    def test_build_grid_from_mesh_empty_mesh(
        self, basic_grid_setting: GridSetting
    ):
        """Test build_grid_from_mesh with empty mesh."""
        # Create an empty mesh
        points = torch.empty((0, 3))
        faces = torch.empty((0, 3), dtype=torch.int32)
        gaussian_curvatures = torch.empty(0)
        empty_mesh = TriangleMesh(
            points=points,
            faces=faces,
            gaussian_curvatures=gaussian_curvatures,
            tree=None,
        )

        forest = Forest(basic_grid_setting)
        grid = forest.build_grid_from_mesh(empty_mesh)

        # Should still return a valid grid
        assert isinstance(grid, Grid)
        assert grid.actual_max_depth >= 0


class TestForestIntegration:
    """Test Forest integration scenarios."""

    def test_forest_complete_workflow(
        self, basic_grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test complete Forest workflow."""
        forest = Forest(basic_grid_setting)

        # Build grid
        grid = forest.build_grid_from_mesh(simple_mesh)

        # Verify internal state
        assert len(forest._nodes) == 1
        assert len(forest._cubes) == 1
        assert forest._actual_max_depth == 0

        # Verify grid properties
        assert grid.domain == forest._bbox
        assert grid.actual_max_depth == forest._actual_max_depth
        assert len(grid.cubes) == len(forest._cubes)

    def test_forest_multiple_depths(
        self, basic_grid_setting: GridSetting, high_curvature_mesh: TriangleMesh
    ):
        """Test Forest with multiple depth levels."""
        forest = Forest(basic_grid_setting)

        _ = forest.build_grid_from_mesh(high_curvature_mesh)

        # Should have multiple depth levels
        assert len(forest._nodes) > 1
        assert len(forest._cubes) > 1

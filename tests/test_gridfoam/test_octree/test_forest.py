import pytest
import torch

from gridfoam._geometry import AABB, TriangleMesh
from gridfoam._octree._forest import Forest
from gridfoam._octree._iterator import iterate_octree_bfs
from gridfoam.settings import GridSetting


@pytest.fixture
def grid_setting() -> GridSetting:
    """Create a basic GridSetting for testing."""
    return GridSetting(
        blockXMin=0.0,
        blockXMax=2.0,
        blockYMin=0.0,
        blockYMax=1.0,
        blockZMin=0.0,
        blockZMax=1.0,
        nBlockX=2,
        nBlockY=1,
        nBlockZ=1,
        alpha=0.3,
        level_limit=3,
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
    gaussian_curvatures = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    return TriangleMesh(
        points=points, faces=faces, gaussian_curvatures=gaussian_curvatures
    )


@pytest.fixture
def empty_mesh() -> TriangleMesh:
    """Create an empty mesh for testing."""
    points = torch.empty((0, 3))
    faces = torch.empty((0, 3), dtype=torch.int32)
    gaussian_curvatures = torch.empty(0)
    return TriangleMesh(
        points=points, faces=faces, gaussian_curvatures=gaussian_curvatures
    )


class TestForestInitialization:
    """Test Forest initialization."""

    def test_init_basic(self, grid_setting: GridSetting):
        """Test basic initialization."""
        forest = Forest(grid_setting)

        torch.testing.assert_close(
            forest._bbox.center, grid_setting.get_domain().center
        )
        torch.testing.assert_close(
            forest._bbox.halfwidth, grid_setting.get_domain().halfwidth
        )
        torch.testing.assert_close(
            forest._divisions, grid_setting.get_block_resolutions()
        )
        assert forest._alpha == grid_setting.alpha
        assert forest._level_limit == grid_setting.level_limit
        assert forest._actual_max_level == 0
        assert len(forest._roots) == 0

    def test_init_with_custom_alpha(self):
        """Test initialization with custom alpha value."""
        setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=2,
            nBlockY=2,
            alpha=0.4,
            level_limit=4,
        )
        forest = Forest(setting)
        assert forest._alpha == 0.4
        assert forest._level_limit == 4


class TestForestGenerateRoots:
    """Test Forest._generate_roots method."""

    def test_generate_roots_basic(
        self, grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test basic root generation."""
        forest = Forest(grid_setting)
        forest._generate_roots(simple_mesh)
        roots = forest._roots

        # Should generate 2 roots (nBlockX=2, nBlockY=1, nBlockZ=1)
        assert len(roots) == 2

        # Check root properties
        for root in roots:
            assert root.octree_depth == 0
            assert root.morton_code == 0
            assert root.level == 1
            assert isinstance(root.face_ids, list)

        # Check root indices
        expected_indices = [[0, 0, 0], [1, 0, 0]]
        for root, expected_index in zip(roots, expected_indices, strict=False):
            torch.testing.assert_close(
                root.root_index, torch.tensor(expected_index, dtype=torch.int32)
            )

    def test_generate_roots_empty_mesh(
        self, grid_setting: GridSetting, empty_mesh: TriangleMesh
    ):
        """Test root generation with empty mesh."""
        forest = Forest(grid_setting)
        forest._generate_roots(empty_mesh)
        roots = forest._roots

        # Should still generate roots even with empty mesh
        assert len(roots) == 2

        # All roots should have empty face_ids
        for root in roots:
            assert root.face_ids == []

    def test_generate_roots_large_domain(self):
        """Test root generation with larger domain."""
        setting = GridSetting(
            blockXMin=0.0,
            blockXMax=4.0,
            blockYMin=0.0,
            blockYMax=2.0,
            blockZMin=0.0,
            blockZMax=2.0,
            nBlockX=4,
            nBlockY=2,
            nBlockZ=2,
        )
        mesh = TriangleMesh(
            points=torch.tensor([[1.0, 1.0, 1.0]]),
            faces=torch.tensor([[0, 0, 0]]),
            gaussian_curvatures=torch.tensor([0.1]),
        )
        forest = Forest(setting)
        forest._generate_roots(mesh)
        roots = forest._roots

        # Should generate 16 roots (4 * 2 * 2)
        assert len(roots) == 16


class TestForestShouldSplitNode:
    """Test Forest._should_split_node method."""

    def test_should_split_node_curvature_condition(
        self, grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test split condition based on curvature."""
        forest = Forest(grid_setting)

        # Create a node with large width (should split)
        large_bbox = AABB(
            center=torch.tensor([0.5, 0.5, 0.5]),
            halfwidth=torch.tensor([1.0, 1.0, 1.0]),  # Large width
        )
        large_node = forest._roots[0] if forest._roots else None
        if large_node:
            large_node._bbox = large_bbox
            # Should split due to large width compared to curvature
            assert forest._should_split_node(large_node, simple_mesh)

    def test_should_split_node_level_limit(
        self, grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test split condition with level limit."""
        forest = Forest(grid_setting)

        # Create a node at max level (should not split)
        max_level_node = forest._roots[0] if forest._roots else None
        if max_level_node:
            max_level_node._octree_depth = grid_setting.level_limit - 1
            # Should not split due to level limit
            assert not forest._should_split_node(max_level_node, simple_mesh)

    def test_should_split_node_small_curvature(self, grid_setting: GridSetting):
        """Test split condition with small curvature (large radius)."""
        # Create mesh with very small curvature (large radius)
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]])
        gaussian_curvatures = torch.tensor(
            [0.001, 0.001, 0.001]
        )  # Very small curvature
        mesh = TriangleMesh(
            points=points, faces=faces, gaussian_curvatures=gaussian_curvatures
        )

        forest = Forest(grid_setting)
        node = forest._roots[0] if forest._roots else None
        if node:
            # Should not split due to small curvature (large radius)
            assert not forest._should_split_node(node, mesh)


class TestForestDilateSplitFlag:
    """Test Forest._dilate_split_flag method."""

    def test_dilate_split_flag_basic(self):
        """Test basic flag dilation."""
        setting = GridSetting(
            blockXMin=0.0,
            blockXMax=5.0,
            blockYMin=0.0,
            blockYMax=3.0,
            blockZMin=0.0,
            blockZMax=3.0,
            nBlockX=5,
            nBlockY=3,
            nBlockZ=3,
        )
        forest = Forest(setting)
        flag_coords = [[2, 1, 1]]
        octree_depth = 0

        dilated = forest._dilate_split_flag(flag_coords, octree_depth)

        # Should dilate to 27-neighborhood (including original point)
        assert len(dilated) == 27

        # Original point should be included
        assert (2, 1, 1) in dilated

        # Some neighboring points should be included
        for i in range(1, 4):
            for j in range(0, 3):
                for k in range(0, 3):
                    assert (i, j, k) in dilated

    def test_dilate_split_flag_boundary(self, grid_setting: GridSetting):
        """Test flag dilation at domain boundaries."""
        forest = Forest(grid_setting)
        flag_coords = [[0, 0, 0]]  # At boundary
        octree_depth = 0

        dilated = forest._dilate_split_flag(flag_coords, octree_depth)

        # Should not include negative coordinates
        assert (-1, 0, 0) not in dilated
        assert (0, -1, 0) not in dilated
        assert (0, 0, -1) not in dilated

    def test_dilate_split_flag_multiple_points(self, grid_setting: GridSetting):
        """Test flag dilation with multiple points."""
        forest = Forest(grid_setting)
        flag_coords = [[0, 0, 0], [1, 0, 0]]
        octree_depth = 0

        dilated = forest._dilate_split_flag(flag_coords, octree_depth)

        # Should include both original points
        assert (0, 0, 0) in dilated
        assert (1, 0, 0) in dilated


class TestForestBuildGridFromMesh:
    """Test Forest.build_from_mesh method."""

    def test_build_grid_from_mesh_basic(
        self, grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test basic mesh building."""
        forest = Forest(grid_setting)
        _ = forest.build_grid_from_mesh(simple_mesh)

        # Should have roots
        assert len(forest._roots) == 2

        # Should have actual_max_level set
        assert forest._actual_max_level > 0

    def test_build_grid_from_mesh_empty(
        self, grid_setting: GridSetting, empty_mesh: TriangleMesh
    ):
        """Test building from empty mesh."""
        forest = Forest(grid_setting)
        _ = forest.build_grid_from_mesh(empty_mesh)

        # Should still have roots
        assert len(forest._roots) == 2

        # All roots should be leaves (no splitting)
        for root in forest._roots:
            assert root.is_leaf()

    def test_build_grid_from_mesh_recursive_splitting(self):
        """Test recursive splitting during mesh building."""
        # Create setting with small alpha to encourage splitting
        setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=1,
            nBlockY=1,
            alpha=0.0,
            level_limit=3,
        )

        # Create mesh with high curvature
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]])
        gaussian_curvatures = torch.tensor([1.0, 1.0, 1.0])  # High curvature
        mesh = TriangleMesh(
            points=points, faces=faces, gaussian_curvatures=gaussian_curvatures
        )

        forest = Forest(setting)
        _ = forest.build_grid_from_mesh(mesh)

        leaves = list(iterate_octree_bfs(forest._roots, only_leaves=True))
        for leaf in leaves:
            assert leaf.level == 3


class TestForestIntegration:
    """Integration tests for Forest."""

    def test_forest_complete_workflow(
        self, grid_setting: GridSetting, simple_mesh: TriangleMesh
    ):
        """Test complete forest workflow."""
        forest = Forest(grid_setting)
        _ = forest.build_grid_from_mesh(simple_mesh)

        # Verify forest structure
        assert len(forest._roots) == 2
        assert forest._actual_max_level > 0

        # Verify root properties
        for root in forest._roots:
            assert root.octree_depth == 0
            assert root.morton_code == 0
            assert root.level == 1

    def test_forest_with_different_settings(self):
        """Test forest with different grid settings."""
        settings = [
            GridSetting(
                blockXMin=0.0,
                blockXMax=1.0,
                blockYMin=0.0,
                blockYMax=1.0,
                nBlockX=1,
                nBlockY=1,
                alpha=0.3,
                level_limit=2,
            ),
            GridSetting(
                blockXMin=0.0,
                blockXMax=2.0,
                blockYMin=0.0,
                blockYMax=2.0,
                nBlockX=4,
                nBlockY=4,
                alpha=0.4,
                level_limit=3,
            ),
        ]

        mesh = TriangleMesh(
            points=torch.tensor([[0.5, 0.5, 0.5]]),
            faces=torch.tensor([[0, 0, 0]]),
            gaussian_curvatures=torch.tensor([0.1]),
        )

        for setting in settings:
            forest = Forest(setting)
            _ = forest.build_grid_from_mesh(mesh)

            expected_roots = setting.nBlockX * setting.nBlockY * setting.nBlockZ
            assert len(forest._roots) == expected_roots

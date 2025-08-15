import pytest
import torch
from trimesh.triangles import bounds_tree

from gridfoam._geometry import AABB, TriangleMesh
from gridfoam._octree._node import OctreeNode


@pytest.fixture
def simple_node() -> OctreeNode:
    """Create a simple OctreeNode for testing."""
    root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
    bbox = AABB(
        min_pt=torch.tensor([0.0, 0.0, 0.0]),
        max_pt=torch.tensor([1.0, 1.0, 1.0]),
    )
    return OctreeNode(
        root_index=root_index,
        bbox=bbox,
        octree_depth=0,
        morton_code=0,
        face_ids=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
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
    gaussian_curvatures = torch.zeros(8)
    tree = bounds_tree(points[faces])
    return TriangleMesh(
        points=points,
        faces=faces,
        gaussian_curvatures=gaussian_curvatures,
        tree=tree,
    )


class TestOctreeNodeInitialization:
    """Test OctreeNode initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        root_index = torch.tensor([1, 2, 3], dtype=torch.int32)
        face_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([1.0, 1.0, 1.0]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=2,
            morton_code=42,
            face_ids=face_ids,
        )

        assert node.bbox == bbox
        assert torch.equal(node.root_index, root_index)
        assert node.octree_depth == 2
        assert node.morton_code == 42
        torch.testing.assert_close(node.face_ids, face_ids)
        assert node.is_leaf is True

    def test_init_empty_face_ids(self):
        """Test initialization with empty face_ids."""
        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        face_ids = torch.tensor([], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([1.0, 1.0, 1.0]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=0,
            morton_code=0,
            face_ids=face_ids,
        )
        torch.testing.assert_close(node.face_ids, face_ids)


class TestOctreeNodeProperties:
    """Test OctreeNode properties."""

    def test_is_leaf_property(self, simple_node: OctreeNode):
        """Test is_leaf property."""
        assert simple_node.is_leaf is True

    def test_is_leaf_property_false(self, simple_node: OctreeNode, simple_mesh: TriangleMesh):
        """Test is_leaf property when node is not a leaf."""
        block_divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        _ = simple_node.split_by_mesh(simple_mesh, block_divisions)
        assert simple_node.is_leaf is False

    def test_root_index_property(self, simple_node: OctreeNode):
        """Test root_index property."""
        expected = torch.tensor([0, 0, 0], dtype=torch.int32)
        assert torch.equal(simple_node.root_index, expected)

    def test_bbox_property(self, simple_node: OctreeNode):
        """Test bbox property."""
        expected_min = torch.tensor([0.0, 0.0, 0.0])
        expected_max = torch.tensor([1.0, 1.0, 1.0])
        assert torch.equal(simple_node.bbox.min, expected_min)
        assert torch.equal(simple_node.bbox.max, expected_max)

    def test_octree_depth_property(self, simple_node: OctreeNode):
        """Test octree_depth property."""
        assert simple_node.octree_depth == 0

    def test_morton_code_property(self, simple_node: OctreeNode):
        """Test morton_code property."""
        assert simple_node.morton_code == 0

    def test_local_index_property(self, simple_node: OctreeNode):
        """Test local_index property."""
        local_index = simple_node.local_index
        assert local_index.shape == (3,)
        assert local_index.dtype == torch.int32
        # For depth 0 and morton_code 0, local_index should be [0, 0, 0]
        assert torch.equal(
            local_index, torch.tensor([0, 0, 0], dtype=torch.int32)
        )

    def test_local_index_property_with_non_zero_morton(self):
        """Test local_index property with non-zero morton code."""
        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([1.0, 1.0, 1.0]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=1,
            morton_code=7,  # [1, 1, 1] in binary
            face_ids=torch.tensor([], dtype=torch.int32),
        )
        local_index = node.local_index
        assert local_index.shape == (3,)
        assert local_index.dtype == torch.int32

    def test_global_index_property(self, simple_node: OctreeNode):
        """Test global_index property."""
        global_index = simple_node.global_index
        assert global_index.shape == (3,)
        assert global_index.dtype == torch.int32
        # For root_index [0,0,0], depth 0, and local_index [0,0,0],
        # global_index should be [0,0,0]
        assert torch.equal(
            global_index, torch.tensor([0, 0, 0], dtype=torch.int32)
        )

    def test_global_index_property_with_non_zero_root(self):
        """Test global_index property with non-zero root index."""
        root_index = torch.tensor([2, 3, 1], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([1.0, 1.0, 1.0]),
        )
        depth = 2
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=depth,
            morton_code=5,
            face_ids=torch.tensor([], dtype=torch.int32),
        )
        global_index = node.global_index
        assert global_index.shape == (3,)
        assert global_index.dtype == torch.int32
        # root_index * 2^octree_depth + local_index
        expected = root_index * (1 << depth) + node.local_index
        assert torch.equal(global_index, expected)

    def test_face_ids_property(self, simple_node: OctreeNode):
        """Test face_ids property."""
        torch.testing.assert_close(simple_node.face_ids, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32))

class TestOctreeNodeMethods:
    """Test OctreeNode methods."""

    def test_calculate_width(self, simple_node: OctreeNode):
        """Test calculate_width method."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        domain_width = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        expected_width = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        torch.testing.assert_close(
            simple_node.calculate_width(divisions, domain_width),
            expected_width,
        )

    def test_root_code(self, simple_node: OctreeNode):
        """Test root_code method."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        expected_root_code = 0
        assert simple_node.root_code(divisions) == expected_root_code

    def test_cube_code(self, simple_node: OctreeNode):
        """Test cube_code method."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        expected_cube_code = 0
        assert simple_node.cube_code(divisions) == expected_cube_code


class TestOctreeNodeSplitByMesh:
    """Test OctreeNode split_by_mesh method."""

    def test_split_by_mesh_basic(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test basic split_by_mesh functionality."""
        block_divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        result = simple_node.split_by_mesh(simple_mesh, block_divisions)
        expected_codes = [
            0x0,
            0x200000000000000,
            0x400000000000000,
            0x600000000000000,
            0x800000000000000,
            0xa00000000000000,
            0xc00000000000000,
            0xe00000000000000,
        ]
        expected_halfwidth = simple_node.bbox.halfwidth / 2

        assert simple_node.is_leaf is False
        assert len(result) == 8

        # Check that all children have correct properties
        child_face_ids = set()
        for i, (code, child) in enumerate(result.items()):
            assert isinstance(child, OctreeNode)
            assert child.octree_depth == simple_node.octree_depth + 1
            assert child.root_code(block_divisions) == 0
            assert child.morton_code == expected_codes[i]
            assert code == expected_codes[i]
            assert code == child.cube_code(block_divisions)
            assert torch.allclose(child.bbox.halfwidth, expected_halfwidth)
            child_face_ids.update(child.face_ids.tolist())
        assert child_face_ids == set(simple_node.face_ids.tolist())

    def test_split_by_mesh_already_split_error(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test that splitting an already split node raises an error."""
        block_divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        _ = simple_node.split_by_mesh(simple_mesh, block_divisions)

        with pytest.raises(ValueError, match="This node is already split"):
            _ = simple_node.split_by_mesh(simple_mesh, block_divisions)


class TestOctreeNodeIntegration:
    """Test OctreeNode integration with other components."""

    def test_node_with_different_root_indices(self):
        """Test nodes with different root indices."""
        root_indices = [
            torch.tensor([0, 0, 0], dtype=torch.int32),
            torch.tensor([1, 0, 0], dtype=torch.int32),
            torch.tensor([0, 1, 0], dtype=torch.int32),
            torch.tensor([1, 1, 1], dtype=torch.int32),
        ]

        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([1.0, 1.0, 1.0]),
        )

        for root_index in root_indices:
            node = OctreeNode(
                root_index=root_index,
                bbox=bbox,
                octree_depth=0,
                morton_code=0,
                face_ids=torch.tensor([], dtype=torch.int32),
            )
            assert torch.equal(node.root_index, root_index)

    def test_node_hierarchy(self, simple_mesh: TriangleMesh):
        """Test node hierarchy creation and traversal."""
        block_divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([1.0, 1.0, 1.0]),
        )
        root_node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=0,
            morton_code=0,
            face_ids=torch.tensor([0, 1, 2], dtype=torch.int32),
        )

        # Split root node
        depth1_nodes =root_node.split_by_mesh(simple_mesh, block_divisions)

        # Split one of the children
        child_to_split = depth1_nodes[0]
        depth2_nodes = child_to_split.split_by_mesh(simple_mesh, block_divisions)

        # Verify hierarchy
        assert not root_node.is_leaf
        assert len(depth1_nodes) == 8
        assert not child_to_split.is_leaf
        assert len(depth2_nodes) == 8

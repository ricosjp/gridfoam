import pytest
import torch
from trimesh.triangles import bounds_tree

from gridfoam._geometry import AABB, TriangleMesh
from gridfoam._octree._node import OctreeNode
from gridfoam.utils.enums import Constants


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
        face_ids=[0, 1, 2, 3, 4, 5, 6, 7],
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
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([1.0, 1.0, 1.0]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=2,
            morton_code=42,
            face_ids=[0, 1, 2, 3],
        )

        assert torch.equal(node.root_index, root_index)
        assert node.bbox == bbox
        assert node.octree_depth == 2
        assert node.morton_code == 42
        assert node.face_ids == [0, 1, 2, 3]
        assert len(node.children) == 0

    def test_init_empty_face_ids(self):
        """Test initialization with empty face_ids."""
        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([1.0, 1.0, 1.0]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=0,
            morton_code=0,
            face_ids=[],
        )

        assert node.face_ids == []


class TestOctreeNodeProperties:
    """Test OctreeNode properties."""

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

    def test_level_property(self, simple_node: OctreeNode):
        """Test level property."""
        assert simple_node.level == 1  # octree_depth + 1

    def test_octree_depth_property(self, simple_node: OctreeNode):
        """Test octree_depth property."""
        assert simple_node.octree_depth == 0

    def test_morton_code_property(self, simple_node: OctreeNode):
        """Test morton_code property."""
        assert simple_node.morton_code == 0

    def test_morton_id_property(self, simple_node: OctreeNode):
        """Test morton_id property."""
        expected_morton_id = 0 << Constants.MORTON_CODE_BIT_LENGTH | 0
        assert simple_node.morton_id == expected_morton_id

    def test_morton_id_property_with_depth(self):
        """Test morton_id property with non-zero depth."""
        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([1.0, 1.0, 1.0]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=5,
            morton_code=42,
            face_ids=[],
        )
        expected_morton_id = 5 << Constants.MORTON_CODE_BIT_LENGTH | 42
        assert node.morton_id == expected_morton_id

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
            face_ids=[],
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
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=2,
            morton_code=5,
            face_ids=[],
        )
        global_index = node.global_index
        assert global_index.shape == (3,)
        assert global_index.dtype == torch.int32
        # root_index * 2^octree_depth + local_index
        expected = root_index * 4 + node.local_index
        assert torch.equal(global_index, expected)

    def test_face_ids_property(self, simple_node: OctreeNode):
        """Test face_ids property."""
        assert simple_node.face_ids == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_children_property(self, simple_node: OctreeNode):
        """Test children property."""
        assert simple_node.children == []


class TestOctreeNodeMethods:
    """Test OctreeNode methods."""

    def test_is_leaf_true(self, simple_node: OctreeNode):
        """Test is_leaf method when node is a leaf."""
        assert simple_node.is_leaf() is True

    def test_is_leaf_false(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test is_leaf method when node has children."""
        simple_node.split_by_mesh(simple_mesh)
        assert simple_node.is_leaf() is False

    def test_has_boundary_true(self, simple_node: OctreeNode):
        """Test has_boundary method when node has faces."""
        assert simple_node.has_boundary() is True

    def test_has_boundary_false(self):
        """Test has_boundary method when node has no faces."""
        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([1.0, 1.0, 1.0]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=0,
            morton_code=0,
            face_ids=[],
        )
        assert node.has_boundary() is False

    def test_has_boundary_false_when_not_leaf(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test has_boundary method when node is not a leaf."""
        simple_node.split_by_mesh(simple_mesh)
        assert simple_node.has_boundary() is False


class TestOctreeNodeSplitByMesh:
    """Test OctreeNode split_by_mesh method."""

    def test_split_by_mesh_basic(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test basic split_by_mesh functionality."""
        assert simple_node.is_leaf() is True
        assert len(simple_node.children) == 0

        simple_node.split_by_mesh(simple_mesh)

        assert simple_node.is_leaf() is False
        assert len(simple_node.children) == 8  # 2^3 = 8 children

        # Check that all children have correct properties
        for child in simple_node.children:
            assert isinstance(child, OctreeNode)
            assert child.octree_depth == simple_node.octree_depth + 1
            assert torch.equal(child.root_index, simple_node.root_index)
            assert isinstance(child.face_ids, list)

    def test_split_by_mesh_child_bboxes(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test that child bounding boxes are correctly calculated."""
        simple_node.split_by_mesh(simple_mesh)

        # Check that child bounding boxes are within parent bounding box
        parent_min = simple_node.bbox.min
        parent_max = simple_node.bbox.max

        for child in simple_node.children:
            child_min = child.bbox.min
            child_max = child.bbox.max

            # Child should be within parent bounds
            assert torch.all(child_min >= parent_min)
            assert torch.all(child_max <= parent_max)

            # Child should have correct halfwidth (parent halfwidth / 2)
            expected_halfwidth = simple_node.bbox.halfwidth / 2
            assert torch.allclose(child.bbox.halfwidth, expected_halfwidth)

    def test_split_by_mesh_morton_codes(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test that child morton codes are correctly assigned."""
        simple_node.split_by_mesh(simple_mesh)

        # Check that all children have different morton codes
        child_codes = [child.morton_code for child in simple_node.children]
        assert len(set(child_codes)) == 8  # All codes should be unique

        # Check that codes are in expected range
        for code in child_codes:
            assert 0 <= code < 2**Constants.MORTON_CODE_BIT_LENGTH

    def test_split_by_mesh_face_distribution(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test that faces are correctly distributed to children."""
        original_face_ids = set(simple_node.face_ids)
        simple_node.split_by_mesh(simple_mesh)

        # Collect all face IDs from children
        child_face_ids = set()
        for child in simple_node.children:
            child_face_ids.update(child.face_ids)

        # All original faces should be present in children
        assert child_face_ids.issubset(original_face_ids)

    def test_split_by_mesh_already_split_error(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test that splitting an already split node raises an error."""
        simple_node.split_by_mesh(simple_mesh)

        with pytest.raises(ValueError, match="This node is already split"):
            simple_node.split_by_mesh(simple_mesh)


class TestOctreeNodeEdgeCases:
    """Test OctreeNode edge cases and error conditions."""

    def test_morton_id_large_depth(self):
        """Test morton_id with large depth."""
        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        bbox = AABB(
            min_pt=torch.tensor([0.0, 0.0, 0.0]),
            max_pt=torch.tensor([1.0, 1.0, 1.0]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=Constants.MAX_OCTREE_DEPTH,
            morton_code=12345,
            face_ids=[],
        )
        # TODO: check

        expected_morton_id = (
            Constants.MAX_OCTREE_DEPTH << Constants.MORTON_CODE_BIT_LENGTH
            | 12345
        )
        assert node.morton_id == expected_morton_id

    def test_global_index_cached_property(self, simple_node: OctreeNode):
        """Test that global_index is cached."""
        # First call should compute the value
        first_call = simple_node.global_index

        # Second call should return cached value
        second_call = simple_node.global_index

        assert torch.equal(first_call, second_call)

        # Check that it's actually the same object (cached)
        assert first_call is second_call


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
                face_ids=[],
            )
            assert torch.equal(node.root_index, root_index)

    def test_node_hierarchy(self, simple_mesh: TriangleMesh):
        """Test node hierarchy creation and traversal."""
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
            face_ids=[0, 1, 2],
        )

        # Split root node
        root_node.split_by_mesh(simple_mesh)

        # Split one of the children
        child_to_split = root_node.children[0]
        child_to_split.split_by_mesh(simple_mesh)

        # Verify hierarchy
        assert not root_node.is_leaf()
        assert len(root_node.children) == 8

        # One child should be split, others should be leaves
        split_children = [
            child for child in root_node.children if not child.is_leaf()
        ]
        leaf_children = [
            child for child in root_node.children if child.is_leaf()
        ]

        assert len(split_children) == 1
        assert len(leaf_children) == 7

        # The split child should have 8 children
        assert len(split_children[0].children) == 8

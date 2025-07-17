import pytest
import torch

from gridfoam._geometry import AABB, TriangleMesh
from gridfoam._octree._iterator import (
    iterate_octree_at_depth,
    iterate_octree_bfs,
    iterate_octree_dfs,
)
from gridfoam._octree._node import OctreeNode


@pytest.fixture
def simple_node() -> OctreeNode:
    """Create a simple OctreeNode for testing."""
    root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
    bbox = AABB(
        center=torch.tensor([0.5, 0.5, 0.5]),
        halfwidth=torch.tensor([0.5, 0.5, 0.5]),
    )
    return OctreeNode(
        root_index=root_index,
        bbox=bbox,
        octree_depth=0,
        morton_code=0,
        face_ids=[0, 1, 2],
    )


@pytest.fixture
def simple_mesh() -> TriangleMesh:
    """Create a simple TriangleMesh for testing."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
        ],
        dtype=torch.int32,
    )
    gaussian_curvatures = torch.zeros(4)
    return TriangleMesh(
        points=points, faces=faces, gaussian_curvatures=gaussian_curvatures
    )


@pytest.fixture
def multi_level_tree(
    simple_node: OctreeNode, simple_mesh: TriangleMesh
) -> OctreeNode:
    """Create a multi-level octree for testing."""
    # Split the root node
    simple_node.split_by_mesh(simple_mesh)

    # Split some of the children to create depth 2
    for child in simple_node.children[:3]:  # Split first 3 children
        child.split_by_mesh(simple_mesh)

    return simple_node


class TestIterateOctreeDFS:
    """Test depth-first search iteration."""

    def test_dfs_multi_level_tree(self, multi_level_tree: OctreeNode):
        """Test DFS iteration on a multi-level tree."""
        nodes = list(iterate_octree_dfs(multi_level_tree))

        # Should visit all nodes: 1 root + 8 children + 3*8 grandchildren = 33 nodes
        assert len(nodes) == 33

        # First node should be the root
        assert nodes[0] == multi_level_tree

        # Check that all nodes are unique
        assert len(set(nodes)) == 33

    def test_dfs_multi_level_tree_only_leaves(
        self, multi_level_tree: OctreeNode
    ):
        """Test DFS iteration on a multi-level tree with only_leaves=True."""
        nodes = list(iterate_octree_dfs(multi_level_tree, only_leaves=True))

        # Should only visit leaf nodes: 5 children + 3*8 grandchildren = 29 leaf nodes
        assert len(nodes) == 29

        # All returned nodes should be leaves
        for node in nodes:
            assert node.is_leaf()

    def test_dfs_order(self, multi_level_tree: OctreeNode):
        """Test that DFS visits nodes in correct order (depth-first)."""
        nodes = list(iterate_octree_dfs(multi_level_tree))

        # Root should be first
        assert nodes[0] == multi_level_tree

        # Children should be visited before grandchildren
        root_children = multi_level_tree.children
        for child in root_children:
            child_index = nodes.index(child)
            # Each child should be visited before its own children
            for grandchild in child.children:
                grandchild_index = nodes.index(grandchild)
                assert grandchild_index > child_index

    def test_dfs_empty_children(self):
        """Test DFS with a node that has no children."""
        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        bbox = AABB(
            center=torch.tensor([0.5, 0.5, 0.5]),
            halfwidth=torch.tensor([0.5, 0.5, 0.5]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=0,
            morton_code=0,
            face_ids=[],
        )

        nodes = list(iterate_octree_dfs(node))
        assert len(nodes) == 1
        assert nodes[0] == node


class TestIterateOctreeBFS:
    """Test breadth-first search iteration."""

    def test_bfs_multi_level_tree(self, multi_level_tree: OctreeNode):
        """Test BFS iteration on a multi-level tree."""
        nodes = list(iterate_octree_bfs([multi_level_tree]))

        # Should visit all nodes: 1 root + 8 children + 3*8 grandchildren = 33 nodes
        assert len(nodes) == 33

        # First node should be the root
        assert nodes[0] == multi_level_tree

        # Check that all nodes are unique
        assert len(set(nodes)) == 33

    def test_bfs_multi_level_tree_only_leaves(
        self, multi_level_tree: OctreeNode
    ):
        """Test BFS iteration on a multi-level tree with only_leaves=True."""
        nodes = list(iterate_octree_bfs([multi_level_tree], only_leaves=True))

        # Should only visit leaf nodes: 5 children + 3*8 grandchildren = 29 leaf nodes
        assert len(nodes) == 29

        # All returned nodes should be leaves
        for node in nodes:
            assert node.is_leaf()

    def test_bfs_order(self, multi_level_tree: OctreeNode):
        """Test that BFS visits nodes in correct order (breadth-first)."""
        nodes = list(iterate_octree_bfs([multi_level_tree]))

        # Root should be first
        assert nodes[0] == multi_level_tree

        # All children should be visited before any grandchildren
        root_children = multi_level_tree.children
        child_indices = [nodes.index(child) for child in root_children]

        # Find the first grandchild index
        first_grandchild_index = None
        for child in root_children:
            for grandchild in child.children:
                grandchild_index = nodes.index(grandchild)
                if (
                    first_grandchild_index is None
                    or grandchild_index < first_grandchild_index
                ):
                    first_grandchild_index = grandchild_index

        # All children should be visited before the first grandchild
        for child_index in child_indices:
            assert child_index < first_grandchild_index

    def test_bfs_multiple_roots(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test BFS with multiple root nodes."""
        # Create a second root node
        root_index2 = torch.tensor([1, 0, 0], dtype=torch.int32)
        bbox2 = AABB(
            center=torch.tensor([1.5, 0.5, 0.5]),
            halfwidth=torch.tensor([0.5, 0.5, 0.5]),
        )
        node2 = OctreeNode(
            root_index=root_index2,
            bbox=bbox2,
            octree_depth=0,
            morton_code=0,
            face_ids=[1, 2, 3],
        )

        # Split both nodes
        simple_node.split_by_mesh(simple_mesh)
        node2.split_by_mesh(simple_mesh)

        nodes = list(iterate_octree_bfs([simple_node, node2]))

        # Should visit all nodes from both trees: 2 roots + 2*8 children = 18 nodes
        assert len(nodes) == 18

        # Both roots should be in the result
        assert simple_node in nodes
        assert node2 in nodes

    def test_bfs_empty_queue(self):
        """Test BFS with an empty initial queue."""
        nodes = list(iterate_octree_bfs([]))
        assert len(nodes) == 0


class TestIterateOctreeAtDepth:
    """Test depth-based iteration."""

    def test_at_depth_multi_level_tree_depth_0(
        self, multi_level_tree: OctreeNode
    ):
        """Test iteration at depth 0 in a multi-level tree."""
        nodes = list(iterate_octree_at_depth([multi_level_tree], depth=0))

        # Should only get the root node at depth 0
        assert len(nodes) == 1
        assert nodes[0] == multi_level_tree

    def test_at_depth_multi_level_tree_depth_1(
        self, multi_level_tree: OctreeNode
    ):
        """Test iteration at depth 1 in a multi-level tree."""
        nodes = list(iterate_octree_at_depth([multi_level_tree], depth=1))

        # Should get all 8 children at depth 1
        assert len(nodes) == 8

        # All nodes should be at depth 1
        for node in nodes:
            assert node.octree_depth == 1

    def test_at_depth_multi_level_tree_depth_2(
        self, multi_level_tree: OctreeNode
    ):
        """Test iteration at depth 2 in a multi-level tree."""
        nodes = list(iterate_octree_at_depth([multi_level_tree], depth=2))

        # Should get all grandchildren at depth 2 (3*8 = 24 nodes)
        assert len(nodes) == 24

        # All nodes should be at depth 2
        for node in nodes:
            assert node.octree_depth == 2

    def test_at_depth_multi_level_tree_depth_1_only_leaves(
        self, multi_level_tree: OctreeNode
    ):
        """Test iteration at depth 1 with only_leaves=True."""
        nodes = list(
            iterate_octree_at_depth(
                [multi_level_tree], depth=1, only_leaves=True
            )
        )

        # Should get only leaf nodes at depth 1 (5 nodes, since 3 were split)
        assert len(nodes) == 5

        # All nodes should be at depth 1 and be leaves
        for node in nodes:
            assert node.octree_depth == 1
            assert node.is_leaf()

    def test_at_depth_multi_level_tree_depth_2_only_leaves(
        self, multi_level_tree: OctreeNode
    ):
        """Test iteration at depth 2 with only_leaves=True."""
        nodes = list(
            iterate_octree_at_depth(
                [multi_level_tree], depth=2, only_leaves=True
            )
        )

        # Should get all leaf nodes at depth 2 (24 nodes, all are leaves)
        assert len(nodes) == 24

        # All nodes should be at depth 2 and be leaves
        for node in nodes:
            assert node.octree_depth == 2
            assert node.is_leaf()

    def test_at_depth_nonexistent_depth(self, multi_level_tree: OctreeNode):
        """Test iteration at a depth that doesn't exist."""
        nodes = list(iterate_octree_at_depth([multi_level_tree], depth=10))

        # Should return no nodes
        assert len(nodes) == 0

    def test_at_depth_multiple_roots(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test depth-based iteration with multiple root nodes."""
        # Create a second root node
        root_index2 = torch.tensor([1, 0, 0], dtype=torch.int32)
        bbox2 = AABB(
            center=torch.tensor([1.5, 0.5, 0.5]),
            halfwidth=torch.tensor([0.5, 0.5, 0.5]),
        )
        node2 = OctreeNode(
            root_index=root_index2,
            bbox=bbox2,
            octree_depth=0,
            morton_code=0,
            face_ids=[1, 2, 3],
        )

        # Split both nodes
        simple_node.split_by_mesh(simple_mesh)
        node2.split_by_mesh(simple_mesh)

        nodes = list(iterate_octree_at_depth([simple_node, node2], depth=1))

        # Should get all children at depth 1 from both trees: 2*8 = 16 nodes
        assert len(nodes) == 16

        # All nodes should be at depth 1
        for node in nodes:
            assert node.octree_depth == 1

    def test_at_depth_empty_queue(self):
        """Test depth-based iteration with an empty initial queue."""
        nodes = list(iterate_octree_at_depth([], depth=0))
        assert len(nodes) == 0


class TestIteratorEdgeCases:
    """Test edge cases and error conditions."""
    def test_dfs_very_deep_tree(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test DFS with a very deep tree."""
        # Create a deep tree by repeatedly splitting
        current_node = simple_node
        for _ in range(3):  # Create 3 levels
            current_node.split_by_mesh(simple_mesh)
            current_node = current_node.children[0]  # Take first child

        # Start from root and iterate
        nodes = list(iterate_octree_dfs(simple_node))

        # Should visit all nodes in the tree
        assert len(nodes) > 0
        # All nodes should be unique
        assert len(set(nodes)) == len(nodes)

    def test_bfs_very_wide_tree(
        self, simple_node: OctreeNode, simple_mesh: TriangleMesh
    ):
        """Test BFS with a very wide tree."""
        # Create a wide tree by splitting all nodes at each level
        simple_node.split_by_mesh(simple_mesh)

        # Split all children
        for child in simple_node.children:
            child.split_by_mesh(simple_mesh)

        nodes = list(iterate_octree_bfs([simple_node]))

        # Should visit all nodes: 1 root + 8 children + 8*8 grandchildren = 73 nodes
        assert len(nodes) == 73

        # All nodes should be unique
        assert len(set(nodes)) == len(nodes)


class TestIteratorIntegration:
    """Test integration between different iterator functions."""

    def test_dfs_bfs_consistency(self, multi_level_tree: OctreeNode):
        """Test that DFS and BFS visit the same nodes (in different order)."""
        dfs_nodes = set(iterate_octree_dfs(multi_level_tree))
        bfs_nodes = set(iterate_octree_bfs([multi_level_tree]))

        # Both should visit the same set of nodes
        assert dfs_nodes == bfs_nodes

    def test_dfs_at_depth_consistency(self, multi_level_tree: OctreeNode):
        """Test that DFS and depth-based iteration are consistent."""
        # Get all nodes at depth 1 using depth-based iteration
        depth_1_nodes = set(
            iterate_octree_at_depth([multi_level_tree], depth=1)
        )

        # Get all nodes at depth 1 using DFS and filtering
        dfs_nodes = set(iterate_octree_dfs(multi_level_tree))
        dfs_depth_1_nodes = {
            node for node in dfs_nodes if node.octree_depth == 1
        }

        # Both should give the same result
        assert depth_1_nodes == dfs_depth_1_nodes

    def test_only_leaves_consistency(self, multi_level_tree: OctreeNode):
        """Test that only_leaves parameter works consistently across all iterators."""
        # Get all leaf nodes using different iterators
        dfs_leaves = set(iterate_octree_dfs(multi_level_tree, only_leaves=True))
        bfs_leaves = set(
            iterate_octree_bfs([multi_level_tree], only_leaves=True)
        )

        # Both should return the same set of leaf nodes
        assert dfs_leaves == bfs_leaves

        # All returned nodes should be leaves
        for node in dfs_leaves:
            assert node.is_leaf()

    def test_empty_tree_consistency(self):
        """Test that all iterators handle empty trees consistently."""
        # Create a node but don't add any children
        root_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        bbox = AABB(
            center=torch.tensor([0.5, 0.5, 0.5]),
            halfwidth=torch.tensor([0.5, 0.5, 0.5]),
        )
        node = OctreeNode(
            root_index=root_index,
            bbox=bbox,
            octree_depth=0,
            morton_code=0,
            face_ids=[],
        )

        # All iterators should return the same result for a single node
        dfs_nodes = list(iterate_octree_dfs(node))
        bfs_nodes = list(iterate_octree_bfs([node]))
        depth_nodes = list(iterate_octree_at_depth([node], depth=0))

        assert dfs_nodes == bfs_nodes
        assert dfs_nodes == depth_nodes
        assert len(dfs_nodes) == 1
        assert dfs_nodes[0] == node

from unittest.mock import Mock

from gridfoam._base._iterator import (
    iter_ghost_from_children_cubes_of,
    iter_ghost_from_parent_cubes_of,
    iter_leaf_cubes_of,
)
from gridfoam.cubion import NodeType, PyOctreeLevel, PyOctreeNode


class TestIterator:
    """Test iterator functions."""

    def test_iter_leaf_cubes_of_empty(self) -> None:
        """
        Test iter_leaf_cubes_of
        with empty octree level.
        """
        # Create mock octree level with no nodes
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.nodes = {}

        # Test iteration over empty level
        result = list(iter_leaf_cubes_of(octree_level))
        assert result == []

    def test_iter_leaf_cubes_of_mixed_types(self) -> None:
        """
        Test iter_leaf_cubes_of
        with mixed node types.
        """
        # Create mock nodes of different types
        leaf = Mock(spec=PyOctreeNode)
        leaf.node_type = NodeType.LEAF

        ghost_parent = Mock(spec=PyOctreeNode)
        ghost_parent.node_type = NodeType.GHOST_FROM_PARENT

        ghost_child = Mock(spec=PyOctreeNode)
        ghost_child.node_type = NodeType.GHOST_FROM_CHILD

        # Create mock octree level
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.nodes = {1: leaf, 2: ghost_parent, 3: ghost_child}

        # Test iteration - should only return leaf nodes
        result = list(iter_leaf_cubes_of(octree_level))
        assert len(result) == 1
        assert leaf in result
        assert ghost_parent not in result
        assert ghost_child not in result

    def test_iter_leaf_cubes_of_no_leaves(self) -> None:
        """
        Test iter_leaf_cubes_of
        with no leaf nodes.
        """
        # Create mock non-leaf nodes
        ghost_parent = Mock(spec=PyOctreeNode)
        ghost_parent.node_type = NodeType.GHOST_FROM_PARENT

        ghost_child = Mock(spec=PyOctreeNode)
        ghost_child.node_type = NodeType.GHOST_FROM_CHILD

        # Create mock octree level
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.nodes = {1: ghost_parent, 2: ghost_child}

        # Test iteration - should return empty list
        result = list(iter_leaf_cubes_of(octree_level))
        assert result == []

    def test_iter_ghost_from_parent_cubes_of_empty(self) -> None:
        """
        Test iter_ghost_from_parent_cubes_of
        with empty octree level.
        """
        # Create mock octree level with no nodes
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.nodes = {}

        # Test iteration over empty level
        result = list(iter_ghost_from_parent_cubes_of(octree_level))
        assert result == []

    def test_iter_ghost_from_parent_cubes_of_mixed_types(self) -> None:
        """
        Test iter_ghost_from_parent_cubes_of
        with mixed node types.
        """
        # Create mock nodes of different types
        leaf = Mock(spec=PyOctreeNode)
        leaf.node_type = NodeType.LEAF

        ghost_parent = Mock(spec=PyOctreeNode)
        ghost_parent.node_type = NodeType.GHOST_FROM_PARENT

        ghost_child = Mock(spec=PyOctreeNode)
        ghost_child.node_type = NodeType.GHOST_FROM_CHILD

        # Create mock octree level
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.nodes = {1: leaf, 2: ghost_parent, 3: ghost_child}

        # Test iteration - should only return ghost from parent nodes
        result = list(iter_ghost_from_parent_cubes_of(octree_level))
        assert len(result) == 1
        assert ghost_parent in result
        assert leaf not in result
        assert ghost_child not in result

    def test_iter_ghost_from_parent_cubes_of_no_ghost_parent(self) -> None:
        """
        Test iter_ghost_from_parent_cubes_of
        with no ghost from parent nodes.
        """
        # Create mock non-ghost-parent nodes
        leaf = Mock(spec=PyOctreeNode)
        leaf.node_type = NodeType.LEAF

        ghost_child = Mock(spec=PyOctreeNode)
        ghost_child.node_type = NodeType.GHOST_FROM_CHILD

        # Create mock octree level
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.nodes = {1: leaf, 2: ghost_child}

        # Test iteration - should return empty list
        result = list(iter_ghost_from_parent_cubes_of(octree_level))
        assert result == []

    def test_iter_ghost_from_children_cubes_of_empty(self) -> None:
        """
        Test iter_ghost_from_children_cubes_of
        with empty octree level.
        """
        # Create mock octree level with no nodes
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.nodes = {}

        # Test iteration over empty level
        result = list(iter_ghost_from_children_cubes_of(octree_level))
        assert result == []

    def test_iter_ghost_from_children_cubes_of_mixed_types(self) -> None:
        """
        Test iter_ghost_from_children_cubes_of
        with mixed node types.
        """
        # Create mock nodes of different types
        leaf = Mock(spec=PyOctreeNode)
        leaf.node_type = NodeType.LEAF

        ghost_parent = Mock(spec=PyOctreeNode)
        ghost_parent.node_type = NodeType.GHOST_FROM_PARENT

        ghost_child = Mock(spec=PyOctreeNode)
        ghost_child.node_type = NodeType.GHOST_FROM_CHILD

        # Create mock octree level
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.nodes = {1: leaf, 2: ghost_parent, 3: ghost_child}

        # Test iteration - should only return ghost from child nodes
        result = list(iter_ghost_from_children_cubes_of(octree_level))
        assert len(result) == 1
        assert ghost_child in result
        assert leaf not in result
        assert ghost_parent not in result

    def test_iter_ghost_from_children_cubes_of_no_ghost_child(self) -> None:
        """
        Test iter_ghost_from_children_cubes_of
        with no ghost from child nodes.
        """
        # Create mock non-ghost-child nodes
        leaf = Mock(spec=PyOctreeNode)
        leaf.node_type = NodeType.LEAF

        ghost_parent = Mock(spec=PyOctreeNode)
        ghost_parent.node_type = NodeType.GHOST_FROM_PARENT

        # Create mock octree level
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.nodes = {1: leaf, 2: ghost_parent}

        # Test iteration - should return empty list
        result = list(iter_ghost_from_children_cubes_of(octree_level))
        assert result == []

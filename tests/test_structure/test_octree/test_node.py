import pytest
import torch

from gridfoam._base import GridTensor, grid_tensor
from gridfoam._structure.octree.aabb import AABB
from gridfoam._structure.octree.node import OctreeNode
from gridfoam.utils.enums import CoordinateType


@pytest.fixture
def simple_octree_node() -> OctreeNode:
    return OctreeNode(
        AABB.from_min_and_max(
            grid_tensor([0.0, 0.0, 0.0], CoordinateType.GLOBAL),
            grid_tensor([1.0, 1.0, 1.0], CoordinateType.GLOBAL),
        ),
        depth=0,
    )


@pytest.fixture
def octree_node_with_indices() -> OctreeNode:
    indices = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]
    )
    return OctreeNode(
        AABB.from_min_and_max(
            grid_tensor([0.0, 0.0, 0.0], CoordinateType.GLOBAL),
            grid_tensor([1.0, 1.0, 1.0], CoordinateType.GLOBAL),
        ),
        depth=1,
        indices=indices,
    )


@pytest.fixture
def octree_node_with_children() -> OctreeNode:
    parent = OctreeNode(
        AABB.from_min_and_max(
            grid_tensor([0.0, 0.0, 0.0], CoordinateType.GLOBAL),
            grid_tensor([1.0, 1.0, 1.0], CoordinateType.GLOBAL),
        ),
        depth=0,
        is_leaf=False,
    )

    children = []
    for i in range(8):
        min_pt = grid_tensor([0.0, 0.0, 0.0], CoordinateType.GLOBAL)
        max_pt = grid_tensor([0.5, 0.5, 0.5], CoordinateType.GLOBAL)

        if i & 1:  # x
            min_pt[0] = 0.5
            max_pt[0] = 1.0
        if i & 2:  # y
            min_pt[1] = 0.5
            max_pt[1] = 1.0
        if i & 4:  # z
            min_pt[2] = 0.5
            max_pt[2] = 1.0

        child = OctreeNode(AABB.from_min_and_max(min_pt, max_pt), depth=1)
        children.append(child)

    parent.children = children
    return parent


def test_initialization(simple_octree_node: OctreeNode):
    assert simple_octree_node.bbox is not None
    assert simple_octree_node.depth == 0
    assert simple_octree_node.indices is None
    assert len(simple_octree_node.children) == 0
    assert len(simple_octree_node.data_dict) == 0
    assert simple_octree_node.is_leaf is True


def test_initialization_with_indices(octree_node_with_indices: OctreeNode):
    assert octree_node_with_indices.indices is not None
    assert octree_node_with_indices.indices.shape == (3, 3)


def test_initialization_with_children(octree_node_with_children: OctreeNode):
    assert octree_node_with_children.is_leaf is False
    assert len(octree_node_with_children.children) == 8
    assert all(
        child is not None for child in octree_node_with_children.children
    )
    assert all(child.depth == 1 for child in octree_node_with_children.children)


def test_origin_property(simple_octree_node: OctreeNode):
    origin = simple_octree_node.origin
    assert isinstance(origin, GridTensor)
    assert torch.all(origin.tensor() == torch.tensor([0.0, 0.0, 0.0]))


def test_center_property(simple_octree_node: OctreeNode):
    center = simple_octree_node.center
    assert isinstance(center, GridTensor)
    assert torch.all(center.tensor() == torch.tensor([0.5, 0.5, 0.5]))


def test_spacing_property(simple_octree_node: OctreeNode):
    spacing = simple_octree_node.spacing
    assert isinstance(spacing, GridTensor)
    assert torch.all(spacing.tensor() == torch.tensor([0.5, 0.5, 0.5]))


# def test_find_node_leaf(simple_octree_node: OctreeNode):
#     position = grid_tensor([0.25, 0.25, 0.25])
#     found_node = simple_octree_node.find_node(position)
#     assert found_node is simple_octree_node


# def test_find_node_with_children(octree_node_with_children: OctreeNode):
#     position1 = grid_tensor([0.25, 0.25, 0.25], coord_type="global")
#     found_node1 = octree_node_with_children.find_node(position1)
#     assert found_node1 is octree_node_with_children.children[0]

#     position2 = grid_tensor([0.75, 0.75, 0.75], coord_type="global")
#     found_node2 = octree_node_with_children.find_node(position2)
#     assert found_node2 is octree_node_with_children.children[7]


# def test_find_node_invalid_position(octree_node_with_children: OctreeNode):
#     position = grid_tensor([[0.25, 0.25], [0.25, 0.25]], coord_type="global")
#     with pytest.raises(ValueError, match="Position must be a 1D tensor"):
#         octree_node_with_children.find_node(position)

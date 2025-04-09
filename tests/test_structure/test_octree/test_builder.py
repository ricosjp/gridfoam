import pathlib
from collections import deque

import numpy as np
import pytest
import pyvista as pv
import torch

from gridfoam._base.tensors import grid_tensor
from gridfoam._structure.octree.aabb import AABB
from gridfoam._structure.octree.builder import Octree
from gridfoam._structure.octree.node import OctreeNode
from gridfoam.utils.enums import CoordinateType


@pytest.fixture
def octree_node_with_depth() -> OctreeNode:
    triangles = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [0, 2, 4],
            [2, 5, 4],
        ]
    )

    node = OctreeNode(
        AABB.from_origin_and_size(
            grid_tensor([1.75, 1.75, 1.75], CoordinateType.GLOBAL),
            grid_tensor([0.25, 0.25, 0.25], CoordinateType.GLOBAL),
        ),
        depth=2,
        indices=torch.from_numpy(triangles),
    )
    return node


@pytest.fixture
def octree() -> Octree:
    """Create an Octree instance for testing."""
    octree = Octree(max_depth=3, triangles_per_leaf=10)
    vertices = np.array(
        [
            [1.75, 1.75, 1.75],
            [1.875, 1.75, 1.75],
            [1.75, 1.875, 1.75],
            [1.875, 1.875, 1.75],
            [1.75, 1.75, 1.875],
            [1.75, 1.875, 1.875],
        ]
    )
    octree.vertices = grid_tensor(vertices, coord_type=CoordinateType.GLOBAL)
    return octree


def test_initialization(octree: Octree):
    """Test Octree initialization."""
    assert octree.max_depth == 3
    assert octree.triangles_per_leaf == 10
    assert octree.actual_depth == 0
    assert octree.root is None


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/stl/bunny.stl"),
    ],
)
def test_build_from_mesh(octree: Octree, file_name: pathlib.Path):
    """Test building Octree from a mesh."""
    pvmesh = pv.read(file_name).extract_surface()
    trimesh = pvmesh.triangulate()
    original_n_triangles = trimesh.regular_faces.shape[0]

    # Build the octree
    octree.build_from_mesh(pvmesh)

    assert octree.root is not None
    assert isinstance(octree.root, OctreeNode)
    assert octree.root.depth == 0

    max_depth = 0
    total_n_triangles = 0
    queue = deque([octree.root])
    while queue:
        node: OctreeNode = queue.popleft()
        if node.is_leaf:
            assert len(node.children) == 0
            if node.depth < octree.max_depth:
                assert node.indices.shape[0] < octree.triangles_per_leaf
            else:
                assert node.depth == octree.max_depth
            max_depth = max(max_depth, node.depth)
            assert node.indices is not None
            total_n_triangles += node.indices.shape[0]
            continue
        assert len(node.children) == 8
        assert node.indices is None
        queue.extend(node.children)
    assert octree.actual_depth == max_depth
    assert total_n_triangles >= original_n_triangles


def test_subdivide(octree: Octree, octree_node_with_depth: OctreeNode):
    """Test subdividing a node."""
    octree.subdivide(octree_node_with_depth)
    assert not octree_node_with_depth.is_leaf
    assert len(octree_node_with_depth.children) == 8
    assert octree_node_with_depth.depth == 2
    assert octree_node_with_depth.indices is None

    # Check that each child is a leaf node
    children = octree_node_with_depth.children
    positions = [
        grid_tensor([1.8, 1.8, 1.8], CoordinateType.GLOBAL),
        grid_tensor([1.9, 1.8, 1.8], CoordinateType.GLOBAL),
        grid_tensor([1.8, 1.9, 1.8], CoordinateType.GLOBAL),
        grid_tensor([1.9, 1.9, 1.8], CoordinateType.GLOBAL),
        grid_tensor([1.8, 1.8, 1.9], CoordinateType.GLOBAL),
        grid_tensor([1.9, 1.8, 1.9], CoordinateType.GLOBAL),
        grid_tensor([1.8, 1.9, 1.9], CoordinateType.GLOBAL),
        grid_tensor([1.9, 1.9, 1.9], CoordinateType.GLOBAL),
    ]
    for child, position in zip(children, positions, strict=True):
        assert child.is_leaf
        assert child.depth == 3
        assert position in child.bbox


@pytest.mark.parametrize(
    "file_name, output_file_name",
    [
        (
            pathlib.Path("tests/data/stl/bunny.stl"),
            pathlib.Path("tests/outputs/octree/test_save.vtkhdf"),
        ),
    ],
)
def test_save(file_name: pathlib.Path, output_file_name: pathlib.Path):
    """Test saving Octree."""
    octree = Octree(max_depth=4)
    pvmesh = pv.read(file_name).extract_surface()

    # Build the octree
    octree.build_from_mesh(pvmesh)

    # Save the octree
    octree.save(output_file_name)


# def test_find_node(octree: Octree):
#     """Test finding a node at a position."""
#     # Create a simple octree with one level of subdivision
#     bbox = AABB(3)
#     bbox.min = grid_tensor([0.0, 0.0, 0.0], coord_type=CoordinateType.GLOBAL)
#     bbox.max = grid_tensor([1.0, 1.0, 1.0], coord_type=CoordinateType.GLOBAL)

#     root = OctreeNode(bbox, depth=0)
#     octree.root = root

#     # Create 8 children
#     children = []
#     for i in range(8):
#         child_bbox = AABB(3)
#         x = 0.0 if i % 2 == 0 else 0.5
#         y = 0.0 if (i // 2) % 2 == 0 else 0.5
#         z = 0.0 if (i // 4) % 2 == 0 else 0.5
#         child_bbox.min = grid_tensor(
#             [x, y, z], coord_type=CoordinateType.GLOBAL
#         )
#         child_bbox.max = grid_tensor(
#             [x + 0.5, y + 0.5, z + 0.5], coord_type=CoordinateType.GLOBAL
#         )
#         child = OctreeNode(child_bbox, depth=1)
#         children.append(child)

#     root.children = children
#     root.is_leaf = False

#     # Test finding nodes at different positions
#     positions = [
#         [0.25, 0.25, 0.25],  # Should be in first child (index 0)
#         [0.75, 0.25, 0.25],  # Should be in second child (index 1)
#         [0.25, 0.75, 0.25],  # Should be in third child (index 2)
#         [0.75, 0.75, 0.25],  # Should be in fourth child (index 3)
#         [0.25, 0.25, 0.75],  # Should be in fifth child (index 4)
#         [0.75, 0.25, 0.75],  # Should be in sixth child (index 5)
#         [0.25, 0.75, 0.75],  # Should be in seventh child (index 6)
#         [0.75, 0.75, 0.75],  # Should be in eighth child (index 7)
#     ]

#     for i, pos in enumerate(positions):
#         position = grid_tensor(pos, coord_type=CoordinateType.GLOBAL)
#         node = octree.find_node(position)
#         assert node == children[i]

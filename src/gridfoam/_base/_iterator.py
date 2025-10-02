from collections.abc import Iterator

from gridfoam.cubion import NodeType, PyOctreeLevel, PyOctreeNode


def iter_leaf_cubes_of(octree_level: PyOctreeLevel) -> Iterator[PyOctreeNode]:
    """Iterate over all leaf cubes in the grid."""
    for cube in octree_level.nodes.values():
        if cube.node_type != NodeType.LEAF:
            continue
        yield cube


def iter_ghost_from_parent_cubes_of(
    octree_level: PyOctreeLevel,
) -> Iterator[PyOctreeNode]:
    """Iterate over all ghost from parent cubes in the grid."""
    for cube in octree_level.nodes.values():
        if cube.node_type != NodeType.GHOST_FROM_PARENT:
            continue
        yield cube


def iter_ghost_from_children_cubes_of(
    octree_level: PyOctreeLevel,
) -> Iterator[PyOctreeNode]:
    """Iterate over all ghost from children cubes in the grid."""
    for cube in octree_level.nodes.values():
        if cube.node_type != NodeType.GHOST_FROM_CHILD:
            continue
        yield cube

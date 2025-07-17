from collections import deque
from collections.abc import Iterator

from gridfoam._octree._node import OctreeNode


def iterate_octree_dfs(
    node: OctreeNode, only_leaves: bool = False
) -> Iterator[OctreeNode]:
    if not only_leaves or node.is_leaf():
        yield node
    for child in node.children:
        yield from iterate_octree_dfs(child, only_leaves)


def iterate_octree_bfs(
    init_queue: list[OctreeNode], only_leaves: bool = False
) -> Iterator[OctreeNode]:
    queue = deque(init_queue)
    while queue:
        node = queue.popleft()
        if not only_leaves or node.is_leaf():
            yield node
        queue.extend(node.children)


def iterate_octree_at_depth(
    init_queue: list[OctreeNode], depth: int, only_leaves: bool = False
) -> Iterator[OctreeNode]:
    queue = deque(init_queue)
    while queue:
        node = queue.popleft()
        if node.octree_depth == depth:
            if not only_leaves or node.is_leaf():
                yield node
        queue.extend(node.children)

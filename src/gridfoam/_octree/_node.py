from __future__ import annotations

import torch
from jaxtyping import Float, Int32

from gridfoam._geometry import AABB, TriangleMesh
from gridfoam.utils.annotated_type import CubeCode
from gridfoam.utils.cube_code import gen_cube_code
from gridfoam.utils.index import ravel_index_3d
from gridfoam.utils.morton import (
    get_child_codes,
    morton_code_to_local_index,
)


class OctreeNode:
    """A node in the octree.
    This class manages information about the hierarchical grid structure.

    Attributes
    ----------
    root_index: Int32[torch.Tensor, " 3"]
        Index of the root that this node belongs to
    bbox: AABB
        Bounding box of this node
    octree_depth: int
        Depth of this node in the octree
    morton_code: int
        Morton code of this node
    face_ids: list[int]
        Indices of the faces that are intersected by this node
    """

    def __init__(
        self,
        root_index: Int32[torch.Tensor, " 3"],
        bbox: AABB,
        octree_depth: int,
        morton_code: int,
        face_ids: list[int],
    ):
        self._bbox = bbox
        self._root_index = root_index
        self._octree_depth = octree_depth
        self._morton_code = morton_code
        self._face_ids = face_ids
        self._is_leaf = True

    def split_by_mesh(
        self,
        mesh: TriangleMesh,
        block_divisions: Int32[torch.Tensor, " 3"],
    ) -> dict[CubeCode, OctreeNode]:
        """
        Split this node into its child nodes.

        Each child will cover a subregion of the current node's bounding box.
        The function determines which faces intersect each child node
        and assigns the corresponding face indices to each child.

        Parameters
        ----------
        mesh : TriangleMesh
            To find the intersecting face IDs for each child node.
        """
        if not self.is_leaf:
            raise ValueError("This node is already split")
        self._is_leaf = False
        child_bboxes = self.bbox.split()
        child_codes = get_child_codes(self.morton_code, self.octree_depth)
        child_octree_depth = self.octree_depth + 1

        nodes = {}
        for child_bbox, child_code in zip(
            child_bboxes, child_codes, strict=True
        ):
            child_face_ids = mesh.find_intersecting_face_ids(child_bbox)
            node = OctreeNode(
                root_index=self.root_index,
                bbox=child_bbox,
                octree_depth=child_octree_depth,
                morton_code=child_code,
                face_ids=child_face_ids,
            )
            nodes[node.cube_code(block_divisions)] = node
        return nodes

    def has_boundary(self) -> bool:
        return len(self.face_ids) > 0

    def calculate_width(
        self,
        block_divisions: Int32[torch.Tensor, " 3"],
        domain_width: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, " 3"]:
        octree_size = 1 << self.octree_depth
        global_divisions = block_divisions * octree_size
        return domain_width / global_divisions

    def root_code(self, block_divisions: Int32[torch.Tensor, " 3"]) -> int:
        """Root code"""
        return ravel_index_3d(self.root_index, block_divisions).item()

    def cube_code(self, block_divisions: Int32[torch.Tensor, " 3"]) -> CubeCode:
        """Cube code"""
        root_code = self.root_code(block_divisions)
        morton_code = self.morton_code
        return gen_cube_code(root_code, morton_code)

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    @property
    def root_index(self) -> Int32[torch.Tensor, " 3"]:
        return self._root_index

    @property
    def bbox(self) -> AABB:
        return self._bbox

    @property
    def octree_depth(self) -> int:
        """The layer number from the octree root"""
        return self._octree_depth

    @property
    def morton_code(self) -> int:
        """Morton code without depth information"""
        return self._morton_code

    @property
    def local_index(self) -> Int32[torch.Tensor, " 3"]:
        """Local index [ix, iy, iz] at given octree depth"""
        return morton_code_to_local_index(self.morton_code, self.octree_depth)

    @property
    def global_index(self) -> Int32[torch.Tensor, " 3"]:
        """Global index [gx, gy, gz] at given octree depth"""
        root_index = self.root_index
        local_index = self.local_index
        octree_size = 1 << self.octree_depth
        global_index = root_index * octree_size + local_index
        return global_index

    @property
    def face_ids(self) -> list[int]:
        return self._face_ids

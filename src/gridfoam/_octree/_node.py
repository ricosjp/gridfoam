from __future__ import annotations

from functools import cached_property

import torch
from jaxtyping import Int32

from gridfoam._geometry import AABB, TriangleMesh, is_intersect_aabb_aabb
from gridfoam.utils.enums import Constants
from gridfoam.utils.index import get_grid_indices
from gridfoam.utils.morton import get_child_codes, get_local_index


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
        self._children: list[OctreeNode] = []

    def split_by_mesh(
        self,
        mesh: TriangleMesh,
    ) -> None:
        """
        Split this node into its child nodes.

        Each child will cover a subregion of the current node's bounding box.
        The function determines which faces intersect each child node
        and assigns the corresponding face indices to each child.

        Parameters
        ----------
        per_face_aabbs : list of AABB
            List of AABBs for each face in the mesh.
            Used to determine which faces intersect with each child node.
        """
        if not self.is_leaf():
            raise ValueError("This node is already split")
        per_face_aabbs = mesh.per_face_aabbs
        divisions = torch.full((3,), 2, dtype=torch.int32)
        child_halfwidth = (self.bbox.max - self.bbox.min) / (2 * divisions)
        child_indices = get_grid_indices(divisions)
        child_codes = get_child_codes(self.morton_code, self.octree_depth)
        child_centers = (
            self.bbox.min + (2 * child_indices + 1) * child_halfwidth
        )
        child_octree_depth = self.octree_depth + 1

        for child_center, child_code in zip(
            child_centers, child_codes, strict=True
        ):
            child_bbox = AABB(child_center, child_halfwidth)
            child_face_ids = [
                fid
                for fid in self.face_ids
                if is_intersect_aabb_aabb(child_bbox, per_face_aabbs[fid])
            ]
            self._children.append(
                OctreeNode(
                    root_index=self.root_index,
                    bbox=child_bbox,
                    octree_depth=child_octree_depth,
                    morton_code=child_code,
                    face_ids=child_face_ids,
                )
            )

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def has_boundary(self) -> bool:
        return self.is_leaf() and len(self.face_ids) > 0

    @property
    def root_index(self) -> Int32[torch.Tensor, " 3"]:
        return self._root_index

    @property
    def bbox(self) -> AABB:
        return self._bbox

    @property
    def level(self) -> int:
        """The layer number from the forest"""
        return self._octree_depth + 1

    @property
    def octree_depth(self) -> int:
        """The layer number from the octree root"""
        return self._octree_depth

    @property
    def morton_code(self) -> int:
        """Morton code without depth information"""
        return self._morton_code

    @property
    def morton_id(self) -> int:
        """Morton code with depth information"""
        depth_code = self.octree_depth << Constants.MORTON_CODE_BIT_LENGTH
        return depth_code | self.morton_code

    @property
    def local_index(self) -> Int32[torch.Tensor, " 3"]:
        """Local index [ix, iy, iz] at given octree depth"""
        return get_local_index(self.morton_code, self.octree_depth)

    @cached_property
    def global_index(self) -> Int32[torch.Tensor, " 3"]:
        """Global index [gx, gy, gz] at given octree depth"""
        root_index = self.root_index
        local_index = self.local_index
        octree_size = torch.tensor(2**self.octree_depth, dtype=torch.int32)
        global_index = root_index * octree_size + local_index
        return global_index

    @property
    def face_ids(self) -> list[int]:
        return self._face_ids

    @property
    def children(self) -> list[OctreeNode]:
        return self._children

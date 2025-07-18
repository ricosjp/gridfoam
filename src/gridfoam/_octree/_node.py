from __future__ import annotations

from functools import cached_property

import torch
from jaxtyping import Float, Int32

from gridfoam._geometry import AABB, TriangleMesh
from gridfoam.utils.enums import Constants
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
        mesh : TriangleMesh
            To find the intersecting face IDs for each child node.
        """
        if not self.is_leaf():
            raise ValueError("This node is already split")
        child_bboxes = self.bbox.split()
        child_codes = get_child_codes(self.morton_code, self.octree_depth)
        child_octree_depth = self.octree_depth + 1

        for child_bbox, child_code in zip(
            child_bboxes, child_codes, strict=True
        ):
            child_face_ids = mesh.find_intersecting_face_ids(child_bbox)
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

    def calculate_width(
        self,
        divisions: Int32[torch.Tensor, " 3"],
        domain_width: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, " 3"]:
        octree_size = 2**self.octree_depth
        total_divisions = divisions * octree_size
        return domain_width / total_divisions

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
        octree_size = 2**self.octree_depth
        global_index = root_index * octree_size + local_index
        return global_index

    @property
    def face_ids(self) -> list[int]:
        return self._face_ids

    @property
    def children(self) -> list[OctreeNode]:
        return self._children

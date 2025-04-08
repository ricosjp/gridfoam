from __future__ import annotations

import torch

from gridfoam._base.tensors import GridTensor
from gridfoam._structure.cube import Cube
from gridfoam._structure.octree.aabb import AABB


class OctreeNode:
    """A node in the octree.
    This class manages information about the hierarchical grid structure.
    The actual data is handled by the Cube class.

    Attributes
    ----------
    bbox: AABB
        Bounding box of the cube
    depth : int
        Depth of the node
    indices : torch.Tensor | None
        Indices of the triangles that are inside the node
    children : List[OctreeNode]
        List of child nodes if this node has been refined
    data_dict : dict[str, Cube]
        Data for the cube
    is_leaf : bool
        Whether the node is a leaf node
    """

    def __init__(
        self,
        bbox: AABB,
        depth: int,
        indices: torch.Tensor | None = None,
        children: list[OctreeNode] | None = None,
        data_dict: dict[str, Cube] | None = None,
        is_leaf: bool = True,
    ):
        self.bbox = bbox
        self.depth = depth
        self.indices = indices
        self.children = children if children is not None else []
        self.data_dict = data_dict if data_dict is not None else {}
        self.is_leaf = is_leaf

    @property
    def origin(self) -> GridTensor:
        return self.bbox.min

    @property
    def center(self) -> GridTensor:
        return self.bbox.center()

    @property
    def spacing(self) -> GridTensor:
        return 0.5 * self.bbox.delta()

    # def find_node(self, position: GridTensor) -> OctreeNode:
    #     #TODO: use morton z order to find the node
    #     if self.is_leaf:
    #         return self
    #     if position.ndim != 1:
    #         raise ValueError("Position must be a 1D tensor")
    #     center = self.center
    #     index = 0
    #     if position[0] >= center[0]:
    #         index += 1
    #     if position[1] >= center[1]:
    #         index += 2
    #     if position[2] >= center[2]:
    #         index += 4
    #     return self.children[index].find_node(position)

    # def add_data(self, field_name: str, dim: int):
    #     """Add data for the cube.

    #     Parameters
    #     ----------
    #     field_name : str
    #         The name of the field to generate
    #     dim : int
    #         The dimension of the field
    #     """
    #     self.data_dict[field_name] = Cube(self.config, dim)

    # def get_values_on(
    #     self, field_name: str, positions: GridTensor
    # ) -> GridTensor:
    #     """Get the values of the field at given positions.

    #     Parameters
    #     ----------
    #     field_name : str
    #         The name of the field to retrieve.
    #     positions : GridTensor
    #         The positions at which to retrieve the values.

    #     Returns
    #     -------
    #     GridTensor
    #         The values of the field at the specified positions.
    #     """
    #     if field_name not in self.data_dict:
    #         raise ValueError(
    #             f"field_name {field_name} is not in the data dictionary"
    #         )

    #     if positions.coord_type == "global":
    #         positions -= self.origin
    #         positions.coord_type = "local"
    #     if positions.coord_type == "local":
    #         positions /= self.spacing
    #         positions -= 0.5
    #         positions.coord_type = "cell"

    #     if not self.contains(positions):
    #         raise ValueError(
    #             "positions are not inside the cube",
    #         )

    #     if positions.coord_type == "cell":
    #         return self.data_dict[field_name].get_values_on(positions)

    #     raise ValueError(
    #         f"Invalid coordinate type: {positions.coord_type}",
    #     )

    # def contains(self, positions: GridTensor) -> bool:
    #     """Check if a given positions are inside the cube.

    #     Parameters
    #     ----------
    #     positions : GridTensor (global or local)
    #         The positions to check

    #     Returns
    #     -------
    #     bool
    #         True if the position is inside the cube, False otherwise
    #     """
    #     if positions.coord_type == "global":
    #         positions = positions - self.origin
    #         positions.coord_type = "local"
    #     if positions.coord_type == "local":
    #         return torch.all(positions >= 0.0) and torch.all(
    #             positions <= self.config.res
    #         )
    #     raise ValueError(
    #         f"Invalid coordinate type: {positions.coord_type}",
    #         "must be 'global' or 'local'",
    #     )

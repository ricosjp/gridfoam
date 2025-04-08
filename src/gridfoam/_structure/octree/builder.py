from __future__ import annotations

import pathlib
from collections import deque

import h5py as h5
import numpy as np
import pyvista as pv
import torch

from gridfoam._base.tensors import grid_tensor
from gridfoam._structure.octree.aabb import AABB
from gridfoam._structure.octree.node import OctreeNode
from gridfoam.utils.enums import CoordinateType


class Octree:
    def __init__(
        self,
        max_depth: int,
        triangles_per_leaf: int = 50,
        actual_depth: int = 0,
    ):
        self.max_depth = max_depth
        self.triangles_per_leaf = triangles_per_leaf
        self.actual_depth = actual_depth
        self.root = None
        self.vertices = None

    def build_from_mesh(self, pvmesh: pv.PolyData):
        if self.root is not None:
            raise ValueError("Octree is already built")
        trimesh = pvmesh.triangulate()
        trimesh_indices = torch.from_numpy(trimesh.regular_faces)
        self.vertices = grid_tensor(
            trimesh.points, coord_type=CoordinateType.GLOBAL
        )
        triangles = self.vertices[trimesh_indices]

        dim = self.vertices.shape[1]
        self.root = OctreeNode(AABB(dim), depth=0)
        self.root.indices = trimesh_indices
        self.root.bbox.push(triangles.reshape((-1, dim)))
        self.root.bbox.scale(1.5)

        # subdivide nodes using Breadth-First Search
        queue = deque([self.root])
        while queue:
            node: OctreeNode = queue.popleft()
            if node.indices.shape[0] < self.triangles_per_leaf:
                self.actual_depth = max(self.actual_depth, node.depth)
                continue
            self.subdivide(node)
            if node.depth == self.max_depth - 1:
                self.actual_depth = self.max_depth
                continue
            queue.extend(node.children)

    def subdivide(self, node: OctreeNode):
        if not node.is_leaf:
            raise ValueError("Cube has already been subdivided")
        node.children = [
            OctreeNode(aabb, depth=node.depth + 1)
            for aabb in node.bbox.subdivide()
        ]
        for child in node.children:
            triangles = self.vertices[node.indices]
            triangles_mask = child.bbox.intersect_triangle(triangles)
            child.indices = node.indices[triangles_mask]

        node.indices = None
        node.is_leaf = False

    # def find_node(self, position: GridTensor) -> OctreeNode:
    #     return self.root.find_node(position)

    def save(self, file_name: pathlib.Path):
        """Save the grid to a file.

        Parameters
        ----------
        file_name : str
            File name to be written. If the parent directory does not exist,
            it will be created.
        overwrite_file : bool
            Whether to overwrite the file if it already exists
        """
        data = self._traverse_octree_for_saving()
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with h5.File(file_name, "w") as f:
            # create hdf root
            vtk_group = f.create_group("VTKHDF", track_order=True)
            vtk_group.attrs["Version"] = [2, 3]
            typeAsASCII = "OverlappingAMR".encode("ascii")
            descriptionAsASCII = "XYZ".encode("ascii")
            vtk_group.attrs.create(
                "Type",
                typeAsASCII,
                dtype=h5.string_dtype("ascii", len(typeAsASCII)),
            )
            vtk_group.attrs["Origin"] = self.root.origin.numpy()
            vtk_group.attrs.create(
                "GridDescription",
                descriptionAsASCII,
                dtype=h5.string_dtype("ascii", len(descriptionAsASCII)),
            )

            data = self._traverse_octree_for_saving()

            for level in range(self.actual_depth):
                key = f"Level{level}"
                level_group = vtk_group.create_group(key)
                level_group.attrs["Spacing"] = data[key]["Spacing"]
                level_group.create_dataset("AMRBox", data=data[key]["AMRBox"])
                _ = level_group.create_group("PointData")
                cell_data = level_group.create_group("CellData")
                _ = level_group.create_group("FieldData")
                cell_data.create_dataset(
                    "cell_values", data=data[key]["CellData"]["cell_values"]
                )

    def _traverse_octree_for_saving(self) -> dict[int, dict]:
        """Traverse the octree for saving.

        Returns
        -------
        dict[int, dict]
        """
        data = {}
        for level in range(self.actual_depth):
            key = f"Level{level}"
            data[key] = {}
            data[key]["AMRBox"] = []
            data[key]["CellData"] = {"cell_values": []}

        queue = deque([self.root])
        while queue:
            node: OctreeNode = queue.popleft()
            if node.is_leaf:
                continue
            level = f"Level{node.depth}"
            if "Spacing" not in data[level]:
                data[level]["Spacing"] = node.spacing.numpy()
            data[level]["AMRBox"].append(self._amr_box(node))
            values = [child.depth for child in node.children]
            data[level]["CellData"]["cell_values"].append(values)

            queue.extend(node.children)

        for level in range(self.actual_depth):
            key = f"Level{level}"
            data[key]["AMRBox"] = np.vstack(data[key]["AMRBox"])
            data[key]["CellData"]["cell_values"] = np.hstack(
                data[key]["CellData"]["cell_values"]
            )
        return data

    def _amr_box(self, node: OctreeNode) -> np.ndarray:
        offset = node.origin - self.root.origin
        min_id = 2 * torch.round(offset / node.bbox.delta()).to(
            dtype=torch.int32
        )
        regions = torch.column_stack((min_id, min_id + 1)).reshape(-1)
        return regions.numpy()

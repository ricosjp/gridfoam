from collections import deque

import torch

from gridfoam._base.grid import Grid
from gridfoam._geometry import AABB, TriangleMesh
from gridfoam._octree._iterator import (
    iterate_octree_at_depth,
    iterate_octree_bfs,
)
from gridfoam._octree._node import OctreeNode
from gridfoam.settings import GridSetting
from gridfoam.utils.flag import dilate_sparse_coords
from gridfoam.utils.index import generate_grid_indices
from gridfoam.utils.log_time import log_time


class Forest:
    def __init__(self, setting: GridSetting):
        self._bbox = setting.get_domain()
        self._divisions = setting.get_block_resolutions()
        self._alpha = setting.alpha
        self._roots: list[OctreeNode] = []
        self._level_limit = setting.level_limit
        self._actual_max_level = 0

    def build_grid_from_mesh(self, mesh: TriangleMesh) -> Grid:
        self._generate_roots(mesh)
        self._recursive_split_by_mesh(mesh)
        level_list = []
        global_index_list = []
        local_code_list = []
        face_ids_list = []
        face_ids_offset_list = []
        for node in iterate_octree_bfs(self._roots, only_leaves=True):
            level_list.append(node.level)
            global_index_list.append(node.global_index.tolist())
            local_code_list.append(node.morton_code)
            offset = len(face_ids_list)
            face_ids_list.extend(node.face_ids)
            face_ids_offset_list.append(offset)
        face_ids_offset_list.append(len(face_ids_list))

        level_array = torch.tensor(level_list, dtype=torch.uint8)
        global_index_array = torch.tensor(global_index_list, dtype=torch.int32)
        local_code_array = torch.tensor(local_code_list, dtype=torch.int64)
        face_ids_array = torch.tensor(face_ids_list, dtype=torch.int32)
        face_ids_offset_array = torch.tensor(
            face_ids_offset_list, dtype=torch.int32
        )
        return Grid(
            domain=self._bbox,
            actual_max_level=self._actual_max_level,
            block_resolutions=self._divisions,
            level=level_array,
            global_index=global_index_array,
            local_code=local_code_array,
            face_ids=face_ids_array,
            face_ids_offset=face_ids_offset_array,
        )

    @log_time
    def _generate_roots(self, mesh: TriangleMesh) -> None:
        """Generate the root nodes of the octree.

        Parameters
        ----------
        mesh : TriangleMesh
            The mesh used to determine which nodes should be split.
        """
        unit_width = self._bbox.width / self._divisions
        root_indices = generate_grid_indices(self._divisions)
        root_mins = self._bbox.min + root_indices * unit_width
        root_maxs = self._bbox.min + (root_indices + 1) * unit_width

        for root_min, root_max, root_index in zip(
            root_mins, root_maxs, root_indices, strict=True
        ):
            root_bbox = AABB(root_min, root_max)
            root_face_ids = mesh.find_intersecting_face_ids(root_bbox)
            self._roots.append(
                OctreeNode(
                    root_index=root_index,
                    bbox=root_bbox,
                    octree_depth=0,
                    morton_code=0,
                    face_ids=root_face_ids,
                )
            )

    @log_time
    def _recursive_split_by_mesh(self, mesh: TriangleMesh) -> None:
        """
        Recursively split octree nodes based on mesh curvature and level limit.

        This method traverses the octree level by level,
        assigning split flags to leaf nodes that meet the splitting condition.
        The split flags are then dilated to neighboring nodes,
        and all flagged nodes are split.
        The process continues until no further nodes are flagged for splitting.

        Parameters
        ----------
        mesh : TriangleMesh
            The mesh used to determine which nodes should be split.
        """
        queue = deque([0])
        while queue:
            depth = queue.popleft()
            # 1. assign split flag to nodes
            assigned_coord_list = []
            for node in iterate_octree_at_depth(
                self._roots, depth, only_leaves=True
            ):
                if not self._should_split_node(node, mesh):
                    continue
                assigned_coord_list.append(node.global_index.tolist())
            if not assigned_coord_list:
                self._actual_max_level = max(self._actual_max_level, depth + 1)
                continue
            # 2. dilate split flag
            dilated_coord_set = self._dilate_split_flag(
                assigned_coord_list, depth
            )
            # 3. split nodes that have split flag
            for node in iterate_octree_at_depth(
                self._roots, depth, only_leaves=True
            ):
                coord = tuple(node.global_index.tolist())
                if coord in dilated_coord_set:
                    node.split_by_mesh(mesh)
            queue.append(depth + 1)

    def _should_split_node(self, node: OctreeNode, mesh: TriangleMesh) -> bool:
        """Check if the node should be split based on curvature and level.

        Parameters
        ----------
        node : OctreeNode
            The octree node to check.
        mesh : TriangleMesh
            The mesh used for curvature calculation.

        Returns
        -------
        bool
            True if the node should be split, False otherwise.
        """
        h = node.calculate_width(self._divisions, self._bbox.width)
        rc = mesh.calculate_radii2_of_curvature(node.face_ids)
        curvature_condition = torch.any(h >= self._alpha * rc).item()
        level_condition = node.level < self._level_limit
        return curvature_condition and level_condition

    def _dilate_split_flag(
        self, flag_coords_list: list[list[int]], octree_depth: int
    ) -> set[tuple[int, int, int]]:
        """Dilate the split flag to
        the 26-neighborhood of the assigned coordinates.

        Parameters
        ----------
        flag_coords_list : list[list[int]]
            List of coordinates to be dilated.
        octree_depth : int
            The depth of the octree.

        Returns
        -------
        set[tuple[int, int, int]]
            Set of coordinates that are dilated.
        """
        octree_size = 2**octree_depth
        bounds = self._divisions * octree_size
        flag_coords = torch.tensor(flag_coords_list, dtype=torch.int32)
        flag_coords = dilate_sparse_coords(flag_coords, bounds)
        split_coord_set = {tuple(coord.tolist()) for coord in flag_coords}
        return split_coord_set

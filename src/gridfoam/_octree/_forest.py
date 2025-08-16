from collections import defaultdict, deque

import torch

from gridfoam._base._cube import Cube
from gridfoam._base.grid import Grid
from gridfoam._geometry import AABB, TriangleMesh
from gridfoam._octree._node import OctreeNode
from gridfoam.settings import GridSetting
from gridfoam.utils.annotated_type import CubeCode
from gridfoam.utils.cube_code import (
    code_to_global_index,
    global_indices_to_codes,
)
from gridfoam.utils.enums import CubeType
from gridfoam.utils.index import generate_grid_indices, neighbor_indices
from gridfoam.utils.log_time import log_time


class Forest:
    def __init__(self, setting: GridSetting):
        self._bbox = setting.get_domain()
        self._block_divisions = setting.get_block_divisions()
        self._alpha = setting.alpha
        self._depth_limit = setting.depth_limit
        self._cube_setting = setting.cube_setting
        self._device = setting.device
        self._actual_max_depth = 0
        self._nodes: dict[int, dict[int, OctreeNode]] = defaultdict(dict)
        self._cubes: dict[int, dict[int, Cube]] = defaultdict(dict)

    def build_grid_from_mesh(self, mesh: TriangleMesh) -> Grid:
        self._generate_roots(mesh)
        self._recursive_split_by_mesh(mesh)
        self._generate_cubes()
        return Grid(
            domain=self._bbox,
            actual_max_depth=self._actual_max_depth,
            block_divisions=self._block_divisions,
            cubes=self._cubes,
            cube_setting=self._cube_setting,
            device=self._device,
        )

    @log_time
    def _generate_roots(self, mesh: TriangleMesh) -> None:
        """Generate the root nodes of the octree.

        Parameters
        ----------
        mesh : TriangleMesh
            The mesh used to determine which nodes should be split.
        """
        unit_width = self._bbox.width / self._block_divisions
        root_indices = generate_grid_indices(self._block_divisions)
        root_mins = self._bbox.min + root_indices * unit_width
        root_maxs = self._bbox.min + (root_indices + 1) * unit_width

        for root_min, root_max, root_index in zip(
            root_mins, root_maxs, root_indices, strict=True
        ):
            root_bbox = AABB(root_min, root_max)
            root_face_ids = mesh.find_intersecting_face_ids(root_bbox)
            node = OctreeNode(
                root_index=root_index,
                bbox=root_bbox,
                octree_depth=0,
                morton_code=0,
                face_ids=root_face_ids,
            )
            self._nodes[0][node.cube_code(self._block_divisions)] = node

    @log_time
    def _recursive_split_by_mesh(self, mesh: TriangleMesh) -> None:
        """
        Recursively split octree nodes based on mesh curvature and depth limit.

        This method traverses the octree depth by depth,
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
            split_coord_list = [
                node.global_index.tolist()
                for node in self._nodes[depth].values()
                if self._should_split_node(node, mesh)
            ]
            if not split_coord_list:
                self._actual_max_depth = max(self._actual_max_depth, depth)
                continue
            # 2. dilate split flag
            dilated_split_code_list = self._dilate_coords(
                split_coord_list, depth
            )
            # 3. split nodes that have split flag
            for code in dilated_split_code_list:
                new_nodes = self._nodes[depth][code].split_by_mesh(
                    mesh, self._block_divisions
                )
                self._nodes[depth + 1].update(new_nodes)
            queue.append(depth + 1)

    @log_time
    def _generate_cubes(self) -> None:
        """Generate cubes from the octree leaf nodes."""
        for depth, nodes_by_depth in self._nodes.items():
            # 1. generate leaf cubes
            leaf_coord_list = []
            for code, node in nodes_by_depth.items():
                if not node.is_leaf:
                    continue
                leaf_coord_list.append(node.global_index.tolist())
                self._cubes[depth][code] = Cube(
                    cube_type=CubeType.LEAF,
                    depth=depth,
                    global_index=node.global_index,
                    face_ids=node.face_ids,
                    device=self._device,
                )
            if len(leaf_coord_list) == 0:
                continue
            # 2. dilate leaf_coord_list
            dilated_code_list = self._dilate_coords(leaf_coord_list, depth)
            ghost_code_set = set(dilated_code_list) - set(
                self._cubes[depth].keys()
            )
            # 3. generate ghost cubes
            for code in ghost_code_set:
                global_index = code_to_global_index(
                    code, self._block_divisions, depth
                )
                cube_type = CubeType.GHOST_FROM_PARENT
                if code in nodes_by_depth:
                    cube_type = CubeType.GHOST_FROM_CHILD
                self._cubes[depth][code] = Cube(
                    cube_type=cube_type,
                    depth=depth,
                    global_index=global_index,
                    face_ids=torch.tensor([], dtype=torch.int32),
                    device=self._device,
                )

    def _should_split_node(self, node: OctreeNode, mesh: TriangleMesh) -> bool:
        """Check if the node should be split based on curvature and depth.

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
        h = node.calculate_width(self._block_divisions, self._bbox.width)
        rc = mesh.calculate_radii2_of_curvature(node.face_ids)
        curvature_condition = torch.any(h >= self._alpha * rc).item()
        depth_condition = node.octree_depth < self._depth_limit
        return curvature_condition and depth_condition

    def _dilate_coords(
        self, coords_list: list[list[int]], octree_depth: int
    ) -> list[CubeCode]:
        """Dilate the coordinates to the 26-neighborhood.

        Parameters
        ----------
        coords_list : list[list[int]]
            List of coordinates to be dilated.
        octree_depth : int
            The depth of the octree.

        Returns
        -------
        list[int]
            List of coordinates that are dilated.
        """
        octree_size = 1 << octree_depth
        bounds = self._block_divisions * octree_size
        coords = torch.tensor(coords_list, dtype=torch.int32)
        coords = neighbor_indices(coords, bounds, include_self=True)
        coords = torch.unique(coords.reshape(-1, 3), dim=0)
        codes = global_indices_to_codes(
            coords, self._block_divisions, octree_depth
        )
        return codes

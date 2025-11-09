from collections.abc import Iterator

import torch
from jaxtyping import Float
from torch.nn import functional as F

from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._field._field import Field
from gridfoam._base._field._grid import (
    NodeType,
    PyOctreeLevel,
    PyOctreeNode,
    RawIndexConversionMode,
)
from gridfoam._base._field._grid import PyGrid as Grid
from gridfoam._base._field._registry import FieldRegistry
from gridfoam._base._io import save_grid
from gridfoam.config import CubeConfig, IoConfig
from gridfoam.utils.enums import FACE_NEIGHBOR_MAP, FieldLayout, FieldRole


class FieldHandle:
    _field_registry: FieldRegistry
    _grid: Grid
    _allocated: bool = False
    _N: int
    _H: int

    @classmethod
    def allocate(
        cls, cube_config: CubeConfig, field_registry: FieldRegistry, grid: Grid
    ) -> None:
        """
        Allocate fields that are declared in the field registry.
        """
        if cls._allocated:
            raise ValueError("Fields already allocated")
        cls._field_registry = field_registry
        cls._grid = grid
        cls._allocated = True
        cls._N = cube_config.interior_width
        cls._H = cube_config.halo_width
        Field.set_cube_config(cube_config)
        for octree_level in grid.octree_levels:
            for cube in octree_level.nodes.values():
                cube.cur = Field()
                cube.old = Field()
                for fd in field_registry.iter_all():
                    cube.old.add_tensor(fd)
                    if fd.role == FieldRole.STATE:
                        cube.cur.add_tensor(fd)

    @classmethod
    def swap_time(cls) -> None:
        """
        Rotate time
        cur → old
        """
        for octree_level in cls._grid.octree_levels:
            for cube in octree_level.nodes.values():
                for fd in cls._field_registry.iter_all():
                    if fd.role != FieldRole.STATE:
                        continue
                    (
                        cube.cur.cells[fd.canonical_name],
                        cube.old.cells[fd.canonical_name],
                    ) = (
                        cube.old.cells[fd.canonical_name],
                        cube.cur.cells[fd.canonical_name],
                    )

    @classmethod
    def sync_halo_at_depth(
        cls, depth: int, fd_list: list[FieldDescriptor]
    ) -> None:
        """
        Synchronize the halo for a given depth.

        Parameters
        ----------
        depth : int
            Depth to synchronize the halo at.
        fd_list : list[FieldDescriptor]
            List of field descriptors to synchronize.
        """
        cell_fd_list = [fd for fd in fd_list if fd.layout == FieldLayout.CELL]
        # face_fd_list = [fd for fd in fd_list if fd.layout == FieldLayout.FACE]

        octree_level = cls._grid.octree_levels[depth]
        bounds = octree_level.bounds
        for cube in cls.iter_leaf_cubes_of(octree_level):
            nbr_codes = cube.cubecode.neighbor_codes(
                depth,
                bounds,
                RawIndexConversionMode.BORDER,
                False,
            )
            for f, (axis, forward) in FACE_NEIGHBOR_MAP.items():
                nbr_code = nbr_codes[f]
                if nbr_code is None:
                    continue
                nbr_cube = octree_level.nodes[nbr_code.value()]
                for fd in cell_fd_list:
                    tgt_cell_tensor = cube.old.cells[fd.canonical_name]
                    nbr_celltensor = nbr_cube.old.cells[fd.canonical_name]
                    nbr_interior_halo = nbr_celltensor.get_interior_halo_along(
                        axis, not forward
                    )
                    tgt_cell_tensor.set_halo_along(
                        axis, forward, nbr_interior_halo
                    )

    @classmethod
    def sync_halo(cls, fd_list: list[FieldDescriptor]) -> None:
        """
        Synchronize the halo for all depths.

        Parameters
        ----------
        fd_list : list[FieldDescriptor]
            List of field descriptors to synchronize.
        """
        for depth in range(cls._grid.max_depth):
            cls.sync_halo_at_depth(depth, fd_list)

    @classmethod
    def sync_ghost_from_parent_at_depth(
        cls, depth: int, fd_list: list[FieldDescriptor]
    ) -> None:
        """
        Synchronize ghost cubes from their parent cubes at a given depth.

        This method extracts data from parent cubes and distributes it to
        ghost cubes that are refined versions of their parent cubes.
        The data is interpolated using trilinear interpolation.

        Parameters
        ----------
        depth : int
            The depth level to synchronize.
        fd_list : list[FieldDescriptor]
            List of field descriptors to synchronize.
        """
        cell_fd_list = [fd for fd in fd_list if fd.layout == FieldLayout.CELL]
        # face_fd_list = [fd for fd in fd_list if fd.layout == FieldLayout.FACE]
        octree_level = cls._grid.octree_levels[depth]
        for cube in cls.iter_ghost_from_parent_cubes_of(octree_level):
            # Get parent cube code and offsets within parent
            parent_code, offsets = cube.cubecode.parent_and_offset_py(depth)
            parent_cube = cls._grid.octree_levels[depth - 1].nodes[
                parent_code.value()
            ]

            # TODO:
            # 1. Piecewise linear reconstruction
            # https://zingale.github.io/comp_astro_tutorial/advection_euler/advection/advection-partIII.html
            # 2. Piecewise Parabolic Method

            for fd in cell_fd_list:
                tgt_cell_tensor = cube.old.cells[fd.canonical_name]
                parent_celltensor = parent_cube.old.cells[fd.canonical_name]
                parent_half_interior = parent_celltensor.get_half_interior(
                    offsets
                )

                # to (1, C, half_width, half_width, half_width)
                parent_half_interior = parent_half_interior.unsqueeze(0)

                # interpolation
                refined_tensor: torch.Tensor = F.interpolate(
                    parent_half_interior,
                    scale_factor=2,
                    mode="trilinear",
                )

                # restore shape to (C, width, width, width)
                refined_tensor = refined_tensor.squeeze(0)

                # assign the refined data to the ghost cube
                tgt_cell_tensor.interior = refined_tensor

    @classmethod
    def sync_ghost_from_parent(cls, fd_list: list[FieldDescriptor]) -> None:
        """
        Synchronize ghost cubes from their parent cubes for all depths.

        This method extracts data from parent cubes and distributes it to
        ghost cubes that are refined versions of their parent cubes.
        The data is interpolated using trilinear interpolation.

        Parameters
        ----------
        fd_list : list[FieldDescriptor]
            List of field descriptors to synchronize.
        """
        for depth in range(cls._grid.max_depth):
            cls.sync_ghost_from_parent_at_depth(depth, fd_list)

    @classmethod
    def sync_ghost_from_children_at_depth(
        cls, depth: int, fd_list: list[FieldDescriptor]
    ) -> None:
        """
        Synchronize ghost cubes from their child cubes at a given depth.

        This method aggregates data from child cubes and distributes it to
        ghost cubes that are coarsened versions of their child cubes.
        The data is coarsened using average pooling
        (2x2x2 cells are averaged into 1 cell).

        Parameters
        ----------
        depth : int
            The depth level to synchronize.
        fd_list : list[FieldDescriptor]
            List of field descriptors to synchronize.
        """
        cell_fd_list = [fd for fd in fd_list if fd.layout == FieldLayout.CELL]
        # face_fd_list = [fd for fd in fd_list if fd.layout == FieldLayout.FACE]
        octree_level = cls._grid.octree_levels[depth]
        for cube in cls.iter_ghost_from_children_cubes_of(octree_level):
            child_codes = cube.cubecode.children(depth)
            for fd in cell_fd_list:
                coarsened_tensors = []
                for child_code in child_codes:
                    child_cube = cls._grid.octree_levels[depth + 1].nodes[
                        child_code.value()
                    ]
                    child_interior = child_cube.old.cells[fd.canonical_name].interior

                    # to (1, C, width, width, width)
                    child_interior = child_interior.unsqueeze(0)

                    # interpolation
                    coarsened_tensor: torch.Tensor = F.avg_pool3d(
                        child_interior,
                        kernel_size=2,
                        stride=2,
                    )

                    # restore shape to (C, width/2, width/2, width/2)
                    coarsened_tensor = coarsened_tensor.squeeze(0)
                    coarsened_tensors.append(coarsened_tensor)

                tgt_cell_tensor = cube.old.cells[fd.canonical_name]
                half_width = tgt_cell_tensor.N // 2
                for i, tensor in enumerate(coarsened_tensors):
                    iz = i // (2 * 2)
                    iy = (i % (2 * 2)) // 2
                    ix = i % 2
                    xs, xe = ix * half_width, (ix + 1) * half_width
                    ys, ye = iy * half_width, (iy + 1) * half_width
                    zs, ze = iz * half_width, (iz + 1) * half_width
                    tgt_cell_tensor.interior[..., zs:ze, ys:ye, xs:xe] = tensor

    @classmethod
    def sync_ghost_from_children(cls, fd_list: list[FieldDescriptor]) -> None:
        """
        Synchronize ghost cubes from their child cubes for all depths.

        This method aggregates data from child cubes
        and distributes it to ghost cubes
        that are coarsened versions of their child cubes.
        The data is coarsened using average pooling
        (2x2x2 cells are averaged into 1 cell).
        """
        for depth in reversed(range(cls._grid.max_depth)):
            cls.sync_ghost_from_children_at_depth(depth, fd_list)

    @classmethod
    def iter_leaf_cubes(cls) -> Iterator[tuple[PyOctreeNode, int]]:
        """
        Iterate over all leaf cubes.
        """
        for octree_level in cls.iter_octree_levels():
            for cube in cls.iter_leaf_cubes_of(octree_level):
                yield cube, octree_level.depth

    @classmethod
    def iter_octree_levels(cls) -> Iterator[PyOctreeLevel]:
        return iter(cls._grid.octree_levels)

    @classmethod
    def iter_leaf_cubes_of(
        cls, octree_level: PyOctreeLevel
    ) -> Iterator[PyOctreeNode]:
        for cube in octree_level.nodes.values():
            if cube.node_type == NodeType.LEAF:
                yield cube

    @classmethod
    def iter_ghost_from_parent_cubes_of(
        cls, octree_level: PyOctreeLevel
    ) -> Iterator[PyOctreeNode]:
        for cube in octree_level.nodes.values():
            if cube.node_type == NodeType.GHOST_FROM_PARENT:
                yield cube

    @classmethod
    def iter_ghost_from_children_cubes_of(
        cls, octree_level: PyOctreeLevel
    ) -> Iterator[PyOctreeNode]:
        for cube in octree_level.nodes.values():
            if cube.node_type == NodeType.GHOST_FROM_CHILD:
                yield cube

    @classmethod
    def get_octree_level_at(cls, depth: int) -> PyOctreeLevel:
        return cls._grid.octree_levels[depth]

    @classmethod
    def get_dx_at(cls, depth: int) -> Float[torch.Tensor, " 3"]:
        octree_level = cls.get_octree_level_at(depth)
        bounds = octree_level.bounds * cls._N
        dx = (cls._grid.domain.upper - cls._grid.domain.lower) / bounds
        return torch.tensor(dx)

    @classmethod
    def save_grid(cls, name: str, io_config: IoConfig) -> None:
        save_grid(cls._grid, io_config, name, cls._N)

from __future__ import annotations

import pathlib
from collections import defaultdict
from dataclasses import dataclass

import pyvista as pv
import torch
import torch.nn.functional as F
import yaml
from jaxtyping import Float32

from gridfoam._base._field_tensor import FieldTensor
from gridfoam.config import Config
from gridfoam.cubion import (
    NodeType,
    PyCubeCode,
    PyGrid,
    RawIndexConversionMode,
    generate_grid_from_polydata,
)
from gridfoam.utils.enums import Direction

TensorSpec = tuple[tuple[int, ...], torch.dtype]


@dataclass
class TensorGrid:
    data: PyGrid
    config: Config
    field_dict: dict[str, TensorSpec]
    mesh: pv.PolyData
    device: torch.device

    @classmethod
    def build(cls, config_path: pathlib.Path) -> TensorGrid:
        # load the config
        with open(config_path) as f:
            raw_yaml = yaml.safe_load(f)
        config = Config.model_validate(raw_yaml)
        device = config.device

        # triangulate the mesh
        mesh = pv.read(config.mesh.file)
        if not isinstance(mesh, pv.PolyData):
            raise ValueError("Mesh must be a PolyData")
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()

        # generate the grid
        data = generate_grid_from_polydata(mesh, str(config_path))

        # by default, add the following fields:
        field_dict = {}
        return cls(
            data=data,
            config=config,
            field_dict=field_dict,
            mesh=mesh,
            device=device,
        )

    @property
    def domain_width(self) -> Float32[torch.Tensor, " 3"]:
        return torch.tensor(
            self.data.domain.upper - self.data.domain.lower,
            device=self.device,
            dtype=torch.float32,
        )

    def update_halo(self) -> None:
        """Update halo cells for all leaf cubes.

        This method updates the halo (ghost) cells for all leaf cubes by
        exchanging boundary data with neighboring cubes.
        It needs a red-black coloring approach
        to avoid race conditions during parallel updates.
        """
        for octree_level in self.data.octree_levels:
            depth = octree_level.depth
            # red-black coloring
            for cube in octree_level.nodes.values():
                code: PyCubeCode = cube.cubecode
                if cube.node_type != NodeType.LEAF:
                    continue
                nbr_codes = code.neighbor_codes(
                    octree_level.depth,
                    octree_level.bounds,
                    RawIndexConversionMode.BORDER,
                    False,
                )

                # x-direction
                ## -x
                nbr_xm_code = nbr_codes[Direction.XM.value]
                if nbr_xm_code is not None:
                    nbr_xm_cube = self.data.octree_levels[depth].nodes[
                        nbr_xm_code.value()
                    ]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_xm_tensor = nbr_xm_cube.field_tensors[name]
                        field_tensor.xm = nbr_xm_tensor.inxp
                ## +x
                nbr_xp_code = nbr_codes[Direction.XP.value]
                if nbr_xp_code is not None:
                    nbr_xp_cube = self.data.octree_levels[depth].nodes[
                        nbr_xp_code.value()
                    ]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_xp_tensor = nbr_xp_cube.field_tensors[name]
                        field_tensor.xp = nbr_xp_tensor.inxm
                # y-direction
                ## -y
                nbr_ym_code = nbr_codes[Direction.YM.value]
                if nbr_ym_code is not None:
                    nbr_ym_cube = self.data.octree_levels[depth].nodes[
                        nbr_ym_code.value()
                    ]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_ym_tensor = nbr_ym_cube.field_tensors[name]
                        field_tensor.ym = nbr_ym_tensor.inyp
                ## +y
                nbr_yp_code = nbr_codes[Direction.YP.value]
                if nbr_yp_code is not None:
                    nbr_yp_cube = self.data.octree_levels[depth].nodes[
                        nbr_yp_code.value()
                    ]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_yp_tensor = nbr_yp_cube.field_tensors[name]
                        field_tensor.yp = nbr_yp_tensor.inym
                # z-direction
                ## -z
                nbr_zm_code = nbr_codes[Direction.ZM.value]
                if nbr_zm_code is not None:
                    nbr_zm_cube = self.data.octree_levels[depth].nodes[
                        nbr_zm_code.value()
                    ]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_zm_tensor = nbr_zm_cube.field_tensors[name]
                        field_tensor.zm = nbr_zm_tensor.inzp
                ## +z
                nbr_zp_code = nbr_codes[Direction.ZP.value]
                if nbr_zp_code is not None:
                    nbr_zp_cube = self.data.octree_levels[depth].nodes[
                        nbr_zp_code.value()
                    ]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_zp_tensor = nbr_zp_cube.field_tensors[name]
                        field_tensor.zp = nbr_zp_tensor.inzm

    def sync_ghost_from_parent(self) -> None:
        """
        Synchronize ghost cubes from their parent cubes.

        This method extracts data from parent cubes
        and distributes it to ghost cubes
        that are refined versions of their parent cubes.
        The data is interpolated  using simple replication
        (each parent cell becomes 2x2x2 child cells).
        """
        half_size = self.config.cube.width // 2

        for octree_level in self.data.octree_levels:
            depth = octree_level.depth
            for cube in octree_level.nodes.values():
                code: PyCubeCode = cube.cubecode
                if cube.node_type != NodeType.GHOST_FROM_PARENT:
                    continue

                # Get parent cube code and offsets within parent
                parent_code, offsets = code.parent_and_offset_py(depth)
                parent_cube = self.data.octree_levels[depth - 1].nodes[
                    parent_code.value()
                ]

                # TODO:
                # 1. Piecewise linear reconstruction
                # https://zingale.github.io/comp_astro_tutorial/advection_euler/advection/advection-partIII.html
                # 2. Piecewise Parabolic Method

                # Calculate the region in parent cube
                # that corresponds to this ghost cube
                # offsets are 0 or 1,
                # indicating which half of the parent cube this ghost occupies
                xs = offsets[0] * half_size
                xe = xs + half_size
                ys = offsets[1] * half_size
                ye = ys + half_size
                zs = offsets[2] * half_size
                ze = zs + half_size

                # Extract data from parent cube and distribute to ghost cube
                for name, parent_tensor in parent_cube.field_tensors.items():
                    # Extract the corresponding region from parent cube
                    parent_region = parent_tensor.interior[zs:ze, ys:ye, xs:xe]

                    # Get shape information
                    region_width, _, _, *extra_shape = parent_region.shape

                    # Interpolate by repeating each cell 2x2x2 times
                    # This creates a refined version
                    refined_tensor = (
                        parent_region.repeat_interleave(2, dim=0)
                        .repeat_interleave(2, dim=1)
                        .repeat_interleave(2, dim=2)
                        .reshape(
                            region_width * 2,
                            region_width * 2,
                            region_width * 2,
                            *extra_shape,
                        )
                    )

                    # Assign the refined data to the ghost cube
                    cube.field_tensors[name].interior = refined_tensor

    def sync_ghost_from_children(self) -> None:
        """Synchronize ghost cubes from their child cubes.

        This method aggregates data from child cubes
        and distributes it to ghost cubes
        that are coarsened versions of their child cubes.
        The data is coarsened using average pooling
        (2x2x2 cells are averaged into 1 cell).
        """
        for octree_level in reversed(self.data.octree_levels):
            depth = octree_level.depth
            for cube in octree_level.nodes.values():
                code: PyCubeCode = cube.cubecode
                if cube.node_type != NodeType.GHOST_FROM_CHILD:
                    continue
                coarsened_tensors_list = defaultdict(list)
                child_codes = code.children(depth)
                # HACK:
                for child_code in child_codes:
                    child_cube = self.data.octree_levels[depth + 1].nodes[
                        child_code.value()
                    ]
                    for name, field_tensor in child_cube.field_tensors.items():
                        interior = field_tensor.interior
                        width, _, _, *extra_shape = interior.shape

                        # to (X, width, width, width)
                        interior_reshaped = interior.reshape(
                            width, width, width, -1
                        )
                        interior_reshaped = interior_reshaped.permute(
                            3, 0, 1, 2
                        )
                        interior_reshaped = interior_reshaped.unsqueeze(0)

                        # interpolation
                        coarsened_tensor = F.avg_pool3d(
                            interior_reshaped, kernel_size=2, stride=2
                        )
                        # to (width/2, width/2, width/2, X)
                        coarsened_tensor = (
                            coarsened_tensor.squeeze(0)
                            .permute(1, 2, 3, 0)
                            .reshape(
                                width // 2, width // 2, width // 2, *extra_shape
                            )
                        )
                        coarsened_tensors_list[name].append(coarsened_tensor)

                block_size = self.config.cube.width // 2
                for name, coarsened_tensors in coarsened_tensors_list.items():
                    for i, tensor in enumerate(coarsened_tensors):
                        iz = i // (2 * 2)
                        iy = (i % (2 * 2)) // 2
                        ix = i % 2
                        xs, xe = ix * block_size, (ix + 1) * block_size
                        ys, ye = iy * block_size, (iy + 1) * block_size
                        zs, ze = iz * block_size, (iz + 1) * block_size
                        cube.field_tensors[name].interior[
                            zs:ze, ys:ye, xs:xe
                        ] = tensor

    def add_field(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """Add a field to the grid.

        Parameters
        ----------
        name : str
            Name of the field.
        shape : tuple[int, ...]
            Shape of the field.
        dtype : torch.dtype
            Data type of the field. example: torch.float32
        """
        if any(x <= 0 for x in shape):
            raise ValueError("All elements must be > 0")
        self.field_dict[name] = (shape, dtype)

    def allocate_field_tensors(self) -> None:
        """Allocate field tensors registered in field_dict for all nodes in the grid."""
        data_width = self.config.cube.width + 2 * self.config.cube.bnd_width
        for octree_level in self.data.octree_levels:
            for node in octree_level.nodes.values():
                node.field_tensors = {}
                for name, spec in self.field_dict.items():
                    shape, dtype = spec
                    shape = (data_width, data_width, data_width, *shape)
                    raw_data = torch.zeros(
                        shape, dtype=dtype, device=self.device
                    )
                    node.field_tensors[name] = FieldTensor(
                        width=self.config.cube.width,
                        bnd=self.config.cube.bnd_width,
                        raw=raw_data,
                    )

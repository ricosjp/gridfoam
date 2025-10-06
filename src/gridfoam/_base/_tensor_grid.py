from __future__ import annotations

import pathlib
from collections import defaultdict
from dataclasses import dataclass

import pyvista as pv
import torch
import torch.nn.functional as F
import yaml
from jaxtyping import Float32

from gridfoam._base._field import Field
from gridfoam._base._iterator import (
    iter_ghost_from_children_cubes_of,
    iter_ghost_from_parent_cubes_of,
    iter_leaf_cubes_of,
)
from gridfoam.config import Config
from gridfoam.cubion import (
    PyCubeCode,
    PyGrid,
    RawIndexConversionMode,
    generate_grid_from_polydata,
)
from gridfoam.utils.enums import FACE_NEIGHBOR_INDEX

TensorSpec = tuple[tuple[int, ...], torch.dtype]


@dataclass
class TensorGrid:
    """
    A tensor grid for finite volume computations with adaptive mesh refinement.

    This class manages a hierarchical grid structure with octree-based
    adaptive mesh refinement (AMR). It provides methods for managing
    cell-centered and face-centered tensor fields across multiple
    refinement levels.

    Parameters
    ----------
    data : PyGrid
        The underlying octree grid data structure.
    config : Config
        Configuration object containing simulation parameters.
    cell_field_dict : dict[str, TensorSpec]
        Dictionary mapping cell field names to their specifications.
    face_field_dict : dict[str, TensorSpec]
        Dictionary mapping face field names to their specifications.
    mesh : pv.PolyData
        The input mesh geometry.
    device : torch.device
        Device where tensors will be allocated.
    """

    data: PyGrid
    config: Config
    cell_field_dict: dict[str, TensorSpec]
    face_field_dict: dict[str, TensorSpec]
    mesh: pv.PolyData
    device: torch.device

    @classmethod
    def build(cls, config_path: pathlib.Path) -> TensorGrid:
        """
        Build a TensorGrid from a configuration file.

        This method loads the configuration, reads the mesh file,
        triangulates it if necessary, and generates the octree grid
        structure from the mesh geometry.

        Parameters
        ----------
        config_path : pathlib.Path
            Path to the YAML configuration file.

        Returns
        -------
        TensorGrid
            A new TensorGrid instance with initialized data structures.

        Raises
        ------
        ValueError
            If the mesh is not a PolyData object.
        """
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

        cell_field_dict = {}
        face_field_dict = {}
        return cls(
            data=data,
            config=config,
            cell_field_dict=cell_field_dict,
            face_field_dict=face_field_dict,
            mesh=mesh,
            device=device,
        )

    @property
    def domain_width(self) -> Float32[torch.Tensor, " 3"]:
        """
        Get the domain width as a tensor.

        Returns
        -------
        Float32[torch.Tensor, " 3"]
            A 3D tensor containing the domain width in each direction.
        """
        return torch.tensor(
            self.data.domain.upper - self.data.domain.lower,
            device=self.device,
            dtype=torch.float32,
        )

    def update_halo_at_depth(self, depth: int) -> None:
        """Update halo cells for a given depth.

        This method updates the halo cells for all leaf cubes at a given depth.
        It needs a red-black coloring approach
        to avoid race conditions during parallel updates.
        """
        octree_level = self.data.octree_levels[depth]
        depth = octree_level.depth
        bounds = octree_level.bounds
        # red-black coloring
        for cube in iter_leaf_cubes_of(octree_level):
            code: PyCubeCode = cube.cubecode
            nbr_codes = code.neighbor_codes(
                depth,
                bounds,
                RawIndexConversionMode.BORDER,
                False,
            )

            for i, f in enumerate(FACE_NEIGHBOR_INDEX):
                axis = i // 2
                forward = bool(i % 2)
                nbr_code = nbr_codes[f]
                if nbr_code is None:
                    continue
                nbr_cube = octree_level.nodes[nbr_code.value()]
                for name, cell_tensor in cube.old.cells.items():
                    nbr_tensor = nbr_cube.old.cells[name]
                    nbr_interior_halo = nbr_tensor.get_interior_halo_along(
                        axis, not forward
                    )
                    cell_tensor.set_halo_along(axis, forward, nbr_interior_halo)

    def update_halo(self) -> None:
        """Update halo cells for all leaf cubes.

        This method updates the halo (ghost) cells for all leaf cubes by
        exchanging boundary data with neighboring cubes.
        It needs a red-black coloring approach
        to avoid race conditions during parallel updates.
        """
        for depth in range(self.data.max_depth):
            self.update_halo_at_depth(depth)

    def sync_ghost_from_parent_at_depth(self, depth: int) -> None:
        """
        Synchronize ghost cubes from their parent cubes at a given depth.

        This method extracts data from parent cubes and distributes it to
        ghost cubes that are refined versions of their parent cubes.
        The data is interpolated using trilinear interpolation.

        Parameters
        ----------
        depth : int
            The depth level to synchronize.
        """
        octree_level = self.data.octree_levels[depth]
        for cube in iter_ghost_from_parent_cubes_of(octree_level):
            code: PyCubeCode = cube.cubecode

            # Get parent cube code and offsets within parent
            parent_code, offsets = code.parent_and_offset_py(depth)
            parent_cube = self.data.octree_levels[depth - 1].nodes[
                parent_code.value()
            ]

            # TODO:
            # 1. Piecewise linear reconstruction
            # https://zingale.github.io/comp_astro_tutorial/advection_euler/advection/advection-partIII.html
            # 2. Piecewise Parabolic Method

            offsets = torch.tensor(
                offsets, device=self.device, dtype=torch.int32
            )

            # Extract data from parent cube and distribute to ghost cube
            for name, parent_tensor in parent_cube.old.cells.items():
                ndim = parent_tensor.ndim
                parent_region = parent_tensor.get_half_interior(offsets)

                # (X2, X1, half_width, half_width, half_width)
                if ndim == 1:
                    parent_region = parent_region.unsqueeze(0)

                # interpolation
                refined_tensor: torch.Tensor = F.interpolate(
                    parent_region,
                    scale_factor=2,
                    mode="trilinear",
                )

                # (X, half_width, half_width, half_width)
                if ndim == 1:
                    refined_tensor = refined_tensor.squeeze(0)

                # Assign the refined data to the ghost cube
                cube.old.cells[name].interior = refined_tensor

    def sync_ghost_from_parent(self) -> None:
        """
        Synchronize ghost cubes from their parent cubes for all depths.

        This method extracts data from parent cubes
        and distributes it to ghost cubes
        that are refined versions of their parent cubes.
        The data is interpolated  using simple replication
        (each parent cell becomes 2x2x2 child cells).
        """
        for depth in range(self.data.max_depth):
            self.sync_ghost_from_parent_at_depth(depth)

    def sync_ghost_from_children_at_depth(self, depth: int) -> None:
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
        """
        octree_level = self.data.octree_levels[depth]
        for cube in iter_ghost_from_children_cubes_of(octree_level):
            code: PyCubeCode = cube.cubecode
            coarsened_tensors_list = defaultdict(list)
            child_codes = code.children(depth)
            # HACK:
            for child_code in child_codes:
                child_cube = self.data.octree_levels[depth + 1].nodes[
                    child_code.value()
                ]
                for name, field_tensor in child_cube.old.cells.items():
                    ndim = field_tensor.ndim
                    interior = field_tensor.interior

                    # (1, X, width, width, width)
                    if ndim == 1:
                        interior = interior.unsqueeze(0)

                    # interpolation
                    coarsened_tensor = F.avg_pool3d(
                        interior, kernel_size=2, stride=2
                    )

                    if ndim == 1:
                        coarsened_tensor = coarsened_tensor.squeeze(0)

                    # (X, width/2, width/2, width/2)
                    coarsened_tensors_list[name].append(coarsened_tensor)

            block_size = self.config.cube.interior_width // 2
            for name, coarsened_tensors in coarsened_tensors_list.items():
                for i, tensor in enumerate(coarsened_tensors):
                    iz = i // (2 * 2)
                    iy = (i % (2 * 2)) // 2
                    ix = i % 2
                    xs, xe = ix * block_size, (ix + 1) * block_size
                    ys, ye = iy * block_size, (iy + 1) * block_size
                    zs, ze = iz * block_size, (iz + 1) * block_size
                    cube.old.cells[name].interior[..., zs:ze, ys:ye, xs:xe] = (
                        tensor
                    )

    def sync_ghost_from_children(self) -> None:
        """Synchronize ghost cubes from their child cubes for all depths.

        This method aggregates data from child cubes
        and distributes it to ghost cubes
        that are coarsened versions of their child cubes.
        The data is coarsened using average pooling
        (2x2x2 cells are averaged into 1 cell).
        """
        for depth in reversed(range(self.data.max_depth)):
            self.sync_ghost_from_children_at_depth(depth)

    def add_cell_field(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """Add a cell field to the grid.

        Parameters
        ----------
        name : str
            Name of the cell field.
        shape : tuple[int, ...]
            Shape of the cell field.
        dtype : torch.dtype
            Data type of the cell field. example: torch.float32
        """
        if any(x <= 0 for x in shape):
            raise ValueError("All elements must be > 0")
        self.cell_field_dict[name] = (shape, dtype)

    def add_face_field(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """
        Add a face field to the grid.

        Parameters
        ----------
        name : str
            Name of the face field.
        shape : tuple[int, ...]
            Shape of the face field.
        dtype : torch.dtype
            Data type of the face field. example: torch.float32
        """
        self.face_field_dict[name] = (shape, dtype)

    def allocate_field_tensors(self) -> None:
        """
        Allocate field tensors
        registered in field_dict for all nodes in the grid.

        This method creates Field instances for each cube in the octree
        and allocates the cell and face tensors according to the
        specifications in cell_field_dict and face_field_dict.
        """
        w_interior = self.config.cube.interior_width
        w_halo = self.config.cube.halo_width
        for octree_level in self.data.octree_levels:
            n_cells_per_node = w_interior**3
            octree_level.n_cells_per_node = n_cells_per_node
            octree_level.n_leaf_cells = (
                octree_level.n_leaf_nodes * n_cells_per_node
            )
            for cube in octree_level.nodes.values():
                cube.cur = Field(w_interior, w_halo, self.device)
                cube.old = Field(w_interior, w_halo, self.device)
                for name, spec in self.cell_field_dict.items():
                    shape, dtype = spec
                    cube.cur.add_cell_tensor(name, shape, dtype)
                    cube.old.add_cell_tensor(name, shape, dtype)
                for name, spec in self.face_field_dict.items():
                    shape, dtype = spec
                    cube.cur.add_face_tensor(name, shape, dtype)
                    cube.old.add_face_tensor(name, shape, dtype)

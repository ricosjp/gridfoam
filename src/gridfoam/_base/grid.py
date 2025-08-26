import pathlib
from collections import defaultdict
from dataclasses import dataclass

import h5py as h5
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int32

from gridfoam._base._cube import Cube
from gridfoam._geometry import AABB
from gridfoam.settings import CubeSetting, FieldDataAttribute
from gridfoam.utils.annotated_type import CubeCode
from gridfoam.utils.cube_code import (
    child_cube_codes,
    code_to_global_index,
    global_indices_to_codes,
    parent_cube_code_and_offsets,
)
from gridfoam.utils.enums import (
    AddressMode,
    CubeType,
    Direction,
    GridCalculationMode,
)
from gridfoam.utils.index import generate_grid_indices, neighbor_indices
from gridfoam.utils.log_time import log_time


@dataclass
class Grid:
    domain: AABB
    actual_max_depth: int
    block_divisions: Int32[torch.Tensor, " 3"]
    cubes: dict[int, dict[CubeCode, Cube]]
    cube_setting: CubeSetting
    field_data_dict: dict[str, FieldDataAttribute]
    device: torch.device = torch.device("cpu")

    def allocate_field_tensors(self) -> None:
        """Allocate field tensors for all cubes and add gradient field.

        This method allocates field tensors for all cubes based on the field_data_dict
        and additionally adds a gradient field tensor for pressure gradient calculations.
        """
        dtype = torch.float32
        for cubes_by_depth in self.cubes.values():
            for cube in cubes_by_depth.values():
                cube.allocate_field_tensors(
                    self.cube_setting, self.field_data_dict
                )
                cube.add_field_tensor(
                    self.cube_setting,
                    "grad_p",
                    FieldDataAttribute(shape=(3,), dtype=dtype),
                )

    def init_p(self) -> None:
        """Initialize pressure field with radial distance from origin.

        This method initializes the pressure field for all cubes by calculating
        the radial distance from the origin for each cell center. The pressure
        is set to the distance from the origin (sqrt(x^2 + y^2 + z^2)).

        After initialization, ghost cells are synchronized from children and
        halo cells are updated.
        """
        dtype = torch.float32
        for depth, cubes_by_depth in self.cubes.items():
            for cube in cubes_by_depth.values():
                # if cube.cube_type != CubeType.LEAF:
                #     continue
                half_spacing = (
                    self.spacing(depth, GridCalculationMode.CELL) * 0.5
                )
                width = self.cube_setting.width
                divisions = torch.tensor([width] * 3, dtype=torch.int32)
                local_cell_indices = generate_grid_indices(divisions).reshape(
                    width, width, width, 3
                )
                global_cell_indices = (
                    cube.global_index[None, None, None, :] * width
                    + local_cell_indices
                )
                global_cell_position = (
                    self.domain.min
                    + (2.0 * global_cell_indices.to(dtype=dtype)[..., :] + 1)
                    * half_spacing
                )
                x = global_cell_position[..., 0]
                y = global_cell_position[..., 1]
                z = global_cell_position[..., 2]
                r = torch.sqrt(x**2 + y**2 + z**2)
                # r = (z-self.domain.center[2])**2
                # r = x+y+z
                cube.field_tensors["p"].interior = r[..., None]
        # propagate to ghost cubes
        self.sync_ghost_from_children()
        # self.sync_ghost_from_parent()

        # update halo cells
        self.update_halo()

    def grad_p(self) -> None:
        """Calculate pressure gradient using finite difference method.

        This method computes the pressure gradient for all cubes using first-order
        finite differences. The gradient is calculated in x, y, and z directions
        using the pressure values at neighboring cells.

        After calculation, ghost cells are synchronized from children and
        halo cells are updated.
        """
        for depth, cubes_by_depth in self.cubes.items():
            for cube in cubes_by_depth.values():
                # if cube.cube_type != CubeType.LEAF:
                #     continue
                spacing = self.spacing(depth, GridCalculationMode.CELL)
                p = cube.field_tensors["p"]
                p_i = p.interior
                p_i_xp = p.raw[
                    p.bnd : -p.bnd, p.bnd : -p.bnd, p.bnd + 1 : -p.bnd + 1
                ]
                p_i_yp = p.raw[
                    p.bnd : -p.bnd, p.bnd + 1 : -p.bnd + 1, p.bnd : -p.bnd
                ]
                p_i_zp = p.raw[
                    p.bnd + 1 : -p.bnd + 1, p.bnd : -p.bnd, p.bnd : -p.bnd
                ]
                dpdx = (p_i_xp - p_i) / spacing[0]
                dpdy = (p_i_yp - p_i) / spacing[1]
                dpdz = (p_i_zp - p_i) / spacing[2]
                cube.field_tensors["grad_p"].interior = torch.cat(
                    [dpdx, dpdy, dpdz], dim=-1
                )
        # propagate to ghost cubes
        self.sync_ghost_from_children()
        # self.sync_ghost_from_parent()

        # update halo cells
        self.update_halo()

    def _calc_amr_box(
        self,
        indices: Int32[torch.Tensor, "n_nodes 3"],
        mode: GridCalculationMode = GridCalculationMode.CELL,
    ) -> Int32[torch.Tensor, "n_nodes 6"]:
        """Calculate AMR box coordinates for given indices.

        Parameters
        ----------
        indices : Int32[torch.Tensor, "n_nodes 3"]
            Global indices of nodes or cells
        mode : GridCalculationMode, default=GridCalculationMode.CELL
            Calculation mode (CELL or NODE)

        Returns
        -------
        Int32[torch.Tensor, "n_nodes 6"]
            AMR box coordinates in format [x_start, x_end, y_start, y_end, z_start, z_end]
        """
        match mode:
            case GridCalculationMode.CELL:
                start = indices * self.cube_setting.width
                end = start + self.cube_setting.width - 1
            case GridCalculationMode.NODE:
                start = indices
                end = indices
            case _:
                raise ValueError(f"Invalid calculation mode: {mode}")
        stacked = torch.stack([start, end], dim=1)
        return stacked.permute(0, 2, 1).reshape(-1, 6)

    def spacing(
        self, depth: int, mode: GridCalculationMode = GridCalculationMode.CELL
    ) -> Float[torch.Tensor, " space_dim"]:
        """Calculate grid spacing for a given depth and mode.

        Parameters
        ----------
        depth : int
            Depth level in the octree
        mode : GridCalculationMode, default=GridCalculationMode.CELL
            Calculation mode (CELL or NODE)

        Returns
        -------
        Float[torch.Tensor, " space_dim"]
            Grid spacing in each dimension
        """
        match mode:
            case GridCalculationMode.CELL:
                octree_size = 1 << depth
                bounds = (
                    self.block_divisions * octree_size * self.cube_setting.width
                )
                return self.domain.width / bounds
            case GridCalculationMode.NODE:
                octree_size = 1 << depth
                bounds = self.block_divisions * octree_size
                return self.domain.width / bounds
            case _:
                raise ValueError(f"Invalid calculation mode: {mode}")

    def update_halo(self) -> None:
        """Update halo cells for all leaf cubes.

        This method updates the halo (ghost) cells for all leaf cubes by
        exchanging boundary data with neighboring cubes.
        It needs a red-black coloring approach
        to avoid race conditions during parallel updates.
        """
        for depth, cubes_by_depth in self.cubes.items():
            # red-black coloring
            for code, cube in cubes_by_depth.items():
                if cube.cube_type != CubeType.LEAF:
                    continue
                global_index = code_to_global_index(
                    code, self.block_divisions, depth
                )
                octree_size = 1 << depth
                bounds = self.block_divisions * octree_size
                nbrs = neighbor_indices(
                    global_index,
                    bounds,
                    address_mode=AddressMode.BORDER,
                )
                nbr_codes = global_indices_to_codes(
                    nbrs, self.block_divisions, depth
                )

                # x-direction
                ## -x
                nbr_xm_code = nbr_codes[Direction.XM.value]
                if nbr_xm_code >= 0:
                    nbr_xm_cube = self.cubes[depth][nbr_xm_code]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_xm_tensor = nbr_xm_cube.field_tensors[name]
                        field_tensor.xm = nbr_xm_tensor.inxp
                ## +x
                nbr_xp_code = nbr_codes[Direction.XP.value]
                if nbr_xp_code >= 0:
                    nbr_xp_cube = self.cubes[depth][nbr_xp_code]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_xp_tensor = nbr_xp_cube.field_tensors[name]
                        field_tensor.xp = nbr_xp_tensor.inxm
                # y-direction
                ## -y
                nbr_ym_code = nbr_codes[Direction.YM.value]
                if nbr_ym_code >= 0:
                    nbr_ym_cube = self.cubes[depth][nbr_ym_code]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_ym_tensor = nbr_ym_cube.field_tensors[name]
                        field_tensor.ym = nbr_ym_tensor.inyp
                ## +y
                nbr_yp_code = nbr_codes[Direction.YP.value]
                if nbr_yp_code >= 0:
                    nbr_yp_cube = self.cubes[depth][nbr_yp_code]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_yp_tensor = nbr_yp_cube.field_tensors[name]
                        field_tensor.yp = nbr_yp_tensor.inym
                # z-direction
                ## -z
                nbr_zm_code = nbr_codes[Direction.ZM.value]
                if nbr_zm_code >= 0:
                    nbr_zm_cube = self.cubes[depth][nbr_zm_code]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_zm_tensor = nbr_zm_cube.field_tensors[name]
                        field_tensor.zm = nbr_zm_tensor.inzp
                ## +z
                nbr_zp_code = nbr_codes[Direction.ZP.value]
                if nbr_zp_code >= 0:
                    nbr_zp_cube = self.cubes[depth][nbr_zp_code]
                    for name, field_tensor in cube.field_tensors.items():
                        nbr_zp_tensor = nbr_zp_cube.field_tensors[name]
                        field_tensor.zp = nbr_zp_tensor.inzm

    def sync_ghost_from_parent(self) -> None:
        """
        Synchronize ghost cubes from their parent cubes.

        This method extracts data from parent cubes and distributes it to ghost cubes
        that are refined versions of their parent cubes. The data is interpolated
        using simple replication (each parent cell becomes 2x2x2 child cells).
        """
        half_size = self.cube_setting.width // 2

        for depth, cubes_by_depth in self.cubes.items():
            for code, cube in cubes_by_depth.items():
                if cube.cube_type != CubeType.GHOST_FROM_PARENT:
                    continue

                # Get parent cube code and offsets within parent
                parent_code, offsets = parent_cube_code_and_offsets(code, depth)
                parent_cube = self.cubes[depth - 1][parent_code]

                # TODO:
                # 1. Piecewise linear reconstruction
                # https://zingale.github.io/comp_astro_tutorial/advection_euler/advection/advection-partIII.html
                # 2. Piecewise Parabolic Method

                # Calculate the region in parent cube that corresponds to this ghost cube
                # offsets are 0 or 1, indicating which half of the parent cube this ghost occupies
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
                    # This creates a refined version where each parent cell becomes 8 child cells
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

        This method aggregates data from child cubes and distributes it to ghost cubes
        that are coarsened versions of their child cubes. The data is coarsened using
        average pooling (2x2x2 cells are averaged into 1 cell).
        """
        for depth, cubes_by_depth in reversed(self.cubes.items()):
            for code, cube in cubes_by_depth.items():
                if cube.cube_type != CubeType.GHOST_FROM_CHILD:
                    continue
                coarsened_tensors_list = defaultdict(list)
                child_codes = child_cube_codes(code, depth)
                # HACK:
                for child_code in child_codes:
                    child_cube = self.cubes[depth + 1][child_code]
                    for name, field_tensor in child_cube.field_tensors.items():
                        interior = field_tensor.interior
                        width, _, _, *extra_shape = interior.shape
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
                        # to (width/2, width/2, width/2, 3)
                        coarsened_tensor = (
                            coarsened_tensor.squeeze(0)
                            .permute(1, 2, 3, 0)
                            .reshape(
                                width // 2, width // 2, width // 2, *extra_shape
                            )
                        )
                        coarsened_tensors_list[name].append(coarsened_tensor)

                block_size = self.cube_setting.width // 2
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

    @log_time
    def save_grid(
        self,
        file_name: pathlib.Path,
        only_leaves: bool = True,
        overwrite_file: bool = True,
    ) -> None:
        """Save the hierarchical grid structure as a VTK HDF file.

        This method saves the grid structure and field data to a VTK HDF file.
        The data is organized by levels, with each level containing spacing,
        AMR box coordinates, and cell data including field tensors.

        Parameters
        ----------
        file_name : pathlib.Path
            File name to be written. If the parent directory does not exist,
            it will be created.
        only_leaves : bool, default=True
            Whether to save only leaf cubes (True) or all cubes (False)
        overwrite_file : bool, default=True
            Whether to overwrite the file if it already exists
        """
        if file_name.exists() and not overwrite_file:
            raise FileExistsError(f"File {file_name} already exists")
        file_name.parent.mkdir(parents=True, exist_ok=True)

        with h5.File(file_name, "w") as f:
            vtk_group = f.create_group("VTKHDF", track_order=True)
            vtk_group.attrs["Version"] = [2, 3]
            typeAsASCII = "OverlappingAMR".encode("ascii")
            descriptionAsASCII = "XYZ".encode("ascii")
            vtk_group.attrs.create(
                "Type",
                typeAsASCII,
                dtype=h5.string_dtype("ascii", len(typeAsASCII)),
            )
            vtk_group.attrs["Origin"] = self.domain.min.cpu().numpy()
            vtk_group.attrs.create(
                "GridDescription",
                descriptionAsASCII,
                dtype=h5.string_dtype("ascii", len(descriptionAsASCII)),
            )

            # level 0
            global_indices0 = []
            cube_types0 = []
            field_data_by_name0 = defaultdict(list)
            cubes_by_depth0 = self.cubes[0]
            for cube in cubes_by_depth0.values():
                if only_leaves and not cube.is_leaf():
                    continue
                global_indices0.append(cube.global_index)
                cube_types0.append(cube.cube_type.value)

                for name, field_tensor in cube.field_tensors.items():
                    _, _, _, *extra_shape = field_tensor.interior.shape
                    data = torch.zeros(
                        (1, *extra_shape), dtype=field_tensor.interior.dtype
                    )
                    field_data_by_name0[name].append(data)

            global_indices0 = torch.stack(global_indices0, dim=0)
            cube_types0 = torch.tensor(cube_types0, dtype=torch.int32)
            depths0 = torch.zeros(global_indices0.shape[0], dtype=torch.int32)

            level_group0 = vtk_group.create_group("Level0")
            level_group0.attrs["Spacing"] = (
                self.spacing(0, GridCalculationMode.NODE).cpu().numpy()
            )
            level_group0.create_dataset(
                "AMRBox",
                data=self._calc_amr_box(
                    global_indices0, GridCalculationMode.NODE
                )
                .cpu()
                .numpy(),
            )
            _ = level_group0.create_group("PointData")
            cell_data0 = level_group0.create_group("CellData")
            _ = level_group0.create_group("FieldData")

            cell_data0.create_dataset(
                "cube_type", data=cube_types0.cpu().numpy()
            )
            cell_data0.create_dataset("depth", data=depths0.cpu().numpy())
            for name, data_list in field_data_by_name0.items():
                concatenated_data = torch.cat(data_list, dim=0).cpu().numpy()
                cell_data0.create_dataset(name, data=concatenated_data)

            # level 1 and above
            for depth, cubes_by_depth in self.cubes.items():
                global_indices = []
                cube_types = []
                field_data_by_name = defaultdict(list)
                for cube in cubes_by_depth.values():
                    if only_leaves and not cube.is_leaf():
                        continue
                    global_indices.append(cube.global_index)
                    cube_types.append(cube.cube_type.value)

                    for name, field_tensor in cube.field_tensors.items():
                        _, _, _, *extra_shape = field_tensor.interior.shape
                        data = field_tensor.interior.reshape(-1, *extra_shape)
                        field_data_by_name[name].append(data)

                global_indices = torch.stack(global_indices, dim=0)
                cube_types = torch.tensor(cube_types, dtype=torch.int32)
                cube_types = torch.repeat_interleave(
                    cube_types, repeats=self.cube_setting.width**3
                )
                depths = torch.full(
                    (cube_types.shape[0],), depth, dtype=torch.int32
                )

                level_group = vtk_group.create_group(f"Level{depth + 1}")
                level_group.attrs["Spacing"] = (
                    self.spacing(depth, GridCalculationMode.CELL).cpu().numpy()
                )
                level_group.create_dataset(
                    "AMRBox",
                    data=self._calc_amr_box(
                        global_indices, GridCalculationMode.CELL
                    )
                    .cpu()
                    .numpy(),
                )
                _ = level_group.create_group("PointData")
                cell_data = level_group.create_group("CellData")
                _ = level_group.create_group("FieldData")

                cell_data.create_dataset(
                    "cube_type", data=cube_types.cpu().numpy()
                )
                cell_data.create_dataset("depth", data=depths.cpu().numpy())
                for name, data_list in field_data_by_name.items():
                    concatenated_data = (
                        torch.cat(data_list, dim=0).cpu().numpy()
                    )
                    cell_data.create_dataset(name, data=concatenated_data)

    @log_time
    def save_structure(
        self,
        file_name: pathlib.Path,
        only_leaves: bool = True,
        overwrite_file: bool = True,
    ) -> None:
        """Save the hierarchical grid structure as a VTK HDF file.

        This method saves only the grid structure (without field data) to a VTK HDF file.
        The data is organized by levels, with each level containing spacing and
        AMR box coordinates.

        Parameters
        ----------
        file_name : pathlib.Path
            File name to be written. If the parent directory does not exist,
            it will be created.
        only_leaves : bool, default=True
            Whether to save only leaf cubes (True) or all cubes (False)
        overwrite_file : bool, default=True
            Whether to overwrite the file if it already exists
        """
        if file_name.exists() and not overwrite_file:
            raise FileExistsError(f"File {file_name} already exists")
        file_name.parent.mkdir(parents=True, exist_ok=True)

        with h5.File(file_name, "w") as f:
            vtk_group = f.create_group("VTKHDF", track_order=True)
            vtk_group.attrs["Version"] = [2, 3]
            typeAsASCII = "OverlappingAMR".encode("ascii")
            descriptionAsASCII = "XYZ".encode("ascii")
            vtk_group.attrs.create(
                "Type",
                typeAsASCII,
                dtype=h5.string_dtype("ascii", len(typeAsASCII)),
            )
            vtk_group.attrs["Origin"] = self.domain.min.cpu().numpy()
            vtk_group.attrs.create(
                "GridDescription",
                descriptionAsASCII,
                dtype=h5.string_dtype("ascii", len(descriptionAsASCII)),
            )
            for depth, cubes_by_depth in self.cubes.items():
                global_indices = []
                cube_types = []
                for cube in cubes_by_depth.values():
                    if only_leaves and not cube.is_leaf():
                        continue
                    global_indices.append(cube.global_index)
                    cube_types.append(cube.cube_type.value)
                global_indices = torch.stack(global_indices, dim=0)
                cube_types = torch.tensor(cube_types, dtype=torch.int32)

                level_group = vtk_group.create_group(f"Level{depth}")
                level_group.attrs["Spacing"] = (
                    self.spacing(depth, GridCalculationMode.NODE).cpu().numpy()
                )
                level_group.create_dataset(
                    "AMRBox",
                    data=self._calc_amr_box(
                        global_indices, GridCalculationMode.NODE
                    )
                    .cpu()
                    .numpy(),
                )
                _ = level_group.create_group("PointData")
                cell_data = level_group.create_group("CellData")
                _ = level_group.create_group("FieldData")

                cell_data.create_dataset(
                    "cube_type", data=cube_types.cpu().numpy()
                )

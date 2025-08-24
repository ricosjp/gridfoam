import pathlib
from dataclasses import dataclass

import h5py as h5
import torch
from jaxtyping import Float, Int32

from gridfoam._base._cube import Cube
from gridfoam._geometry import AABB
from gridfoam.settings import CubeSetting, FieldDataAttribute
from gridfoam.utils.annotated_type import CubeCode
from gridfoam.utils.enums import GridCalculationMode
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
        for cubes_by_depth in self.cubes.values():
            for cube in cubes_by_depth.values():
                cube.allocate_field_tensors(self.cube_setting, self.field_data_dict)

    def _calc_amr_box(
        self,
        indices: Int32[torch.Tensor, "n_nodes 3"],
        mode: GridCalculationMode = GridCalculationMode.CELL,
    ) -> Int32[torch.Tensor, "n_nodes 6"]:
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

    @log_time
    def save_grid(
        self,
        file_name: pathlib.Path,
        only_leaves: bool = True,
        overwrite_file: bool = True,
    ) -> None:
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
            cubes_by_depth0 = self.cubes[0]
            global_indices = []
            cube_types = []
            for cube in cubes_by_depth0.values():
                if only_leaves and not cube.is_leaf():
                    continue
                global_indices.append(cube.global_index)
                cube_types.append(cube.cube_type.value)
            global_indices = torch.stack(global_indices, dim=0)
            cube_types = torch.tensor(cube_types, dtype=torch.int32)

            level_group = vtk_group.create_group("Level0")
            level_group.attrs["Spacing"] = (
                self.spacing(0, GridCalculationMode.NODE).cpu().numpy()
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

            cell_data.create_dataset("cube_type", data=cube_types.cpu().numpy())

            # level 1 and above
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
                cube_types = torch.repeat_interleave(
                    cube_types, repeats=self.cube_setting.width**3
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

    @log_time
    def save_structure(
        self,
        file_name: pathlib.Path,
        only_leaves: bool = True,
        overwrite_file: bool = True,
    ) -> None:
        """Save the hierarchical grid structure as a VTK HDF file.

        Parameters
        ----------
        file_name : str
            File name to be written. If the parent directory does not exist,
            it will be created.
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

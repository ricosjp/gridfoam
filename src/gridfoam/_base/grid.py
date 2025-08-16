import pathlib
from collections import defaultdict
from dataclasses import dataclass

import h5py as h5
import torch
from jaxtyping import Int32

from gridfoam._base._cube import Cube
from gridfoam._geometry import AABB
from gridfoam.settings import CubeSetting
from gridfoam.utils.annotated_type import CubeCode
from gridfoam.utils.log_time import log_time


@dataclass
class Grid:
    domain: AABB
    actual_max_depth: int
    block_divisions: Int32[torch.Tensor, " 3"]
    cubes: dict[int, dict[CubeCode, Cube]]
    cube_setting: CubeSetting
    device: torch.device = torch.device("cpu")

    def _calc_amr_box(
        self, indices: Int32[torch.Tensor, "n_nodes 3"]
    ) -> Int32[torch.Tensor, "n_nodes 6"]:
        n_cell = 0
        stacked = torch.stack([indices, indices + n_cell], dim=1)
        return stacked.permute(0, 2, 1).reshape(-1, 6)

    @log_time
    def save_structure(
        self, file_name: pathlib.Path, overwrite_file: bool = True
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

        data = defaultdict(dict)
        for depth, cubes in self.cubes.items():
            key = f"Level{depth}"
            n_nodes = len(cubes)
            global_indices = torch.stack(
                [cube.global_index for cube in cubes.values()], dim=0
            )
            cube_types = torch.tensor(
                [cube.cube_type.value for cube in cubes.values()], dtype=torch.int32
            )
            depth_data = torch.full((n_nodes,), depth, dtype=torch.int32)
            octree_size = 1 << depth
            bounds = self.block_divisions * octree_size
            spacing = self.domain.width / bounds
            data[key]["Spacing"] = spacing.cpu().numpy()
            data[key]["AMRBox"] = (
                self._calc_amr_box(global_indices).cpu().numpy()
            )
            data[key]["CellData"] = {
                "depth": depth_data.cpu().numpy(),
                "cube_type": cube_types.cpu().numpy(),
            }

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
            for key, depth_data in data.items():
                level_group = vtk_group.create_group(key)
                level_group.attrs["Spacing"] = depth_data["Spacing"]
                level_group.create_dataset("AMRBox", data=depth_data["AMRBox"])
                _ = level_group.create_group("PointData")
                cell_data = level_group.create_group("CellData")
                _ = level_group.create_group("FieldData")
                for field_name, field_data in depth_data["CellData"].items():
                    cell_data.create_dataset(field_name, data=field_data)


import pathlib
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import h5py as h5
import torch
from jaxtyping import Int32, Int64, UInt8

from gridfoam._base.field_data import FieldData
from gridfoam._geometry import AABB
from gridfoam.settings import CubeSetting
from gridfoam.utils.log_time import log_time


@dataclass
class Grid:
    domain: AABB
    actual_max_level: int
    block_resolutions: Int32[torch.Tensor, " 3"]
    level: UInt8[torch.Tensor, " n_nodes"]
    global_index: Int32[torch.Tensor, "n_nodes 3"]
    local_code: Int64[torch.Tensor, " n_nodes"]
    face_ids: Int32[torch.Tensor, " n_ids"]
    face_ids_offset: Int32[torch.Tensor, " n_nodes_plus_1"]
    cube_setting: CubeSetting
    field_data: FieldData | None = None
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        self.n_nodes = self.level.shape[0]
        field_data = FieldData(self.device)
        field_data.add_field("U", (self.n_nodes, 3), torch.float32)
        field_data.add_field("p", (self.n_nodes,), torch.float32)
        self.field_data = field_data

    def add_field(
        self, field_name: str, shape: tuple, dtype: torch.dtype
    ) -> None:
        shape = (self.n_nodes, *shape)
        self.field_data.add_field(field_name, shape, dtype)

    def get_neighbor_indices(
        self, level: int, global_index: Int32[torch.Tensor, "n_target_nodes 3"]
    ) -> Int32[torch.Tensor, "n_target_nodes 26 3"]:
        bounds = self.block_resolutions * 2 ** (level - 1)
        D, H, W = bounds

        # 26-neighborhood offsets
        offsets = torch.tensor(
            list(product([-1, 0, 1], repeat=3)), dtype=torch.int32
        )
        offsets = offsets[(offsets != 0).any(dim=1)]  # exclude (0,0,0)

        # (N, _, 3) + (26, 3) -> (N, 26, 3)
        neighbors = global_index[:, None, :] + offsets[None, :, :]

        # Assign -1 to coordinates outside the bounds
        mask = (
            (neighbors[:, :, 0] >= 0)
            & (neighbors[:, :, 0] < D)
            & (neighbors[:, :, 1] >= 0)
            & (neighbors[:, :, 1] < H)
            & (neighbors[:, :, 2] >= 0)
            & (neighbors[:, :, 2] < W)
        )
        neighbors[~mask] = -1
        return neighbors

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
        for level in range(1, self.actual_max_level + 1):
            key = f"Level{level - 1}"
            mask = self.level == level
            global_index = self.global_index[mask]  # (n_nodes, 3)
            n_nodes = global_index.shape[0]
            level_data = torch.full((n_nodes,), level, dtype=torch.float32)
            octree_size = torch.tensor(2 ** (level - 1), dtype=torch.int32)
            resolution = self.block_resolutions * octree_size
            spacing = self.domain.width / resolution
            data[key]["Spacing"] = spacing.cpu().numpy()
            data[key]["AMRBox"] = self._calc_amr_box(global_index).cpu().numpy()
            data[key]["CellData"] = {"level": level_data.cpu().numpy()}
            for field_name, field_data in self.field_data.items():
                data[key]["CellData"][field_name] = field_data.cpu().numpy()

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

            for level in range(1, self.actual_max_level + 1):
                key = f"Level{level - 1}"
                level_group = vtk_group.create_group(key)
                level_group.attrs["Spacing"] = data[key]["Spacing"]
                level_group.create_dataset("AMRBox", data=data[key]["AMRBox"])
                _ = level_group.create_group("PointData")
                cell_data = level_group.create_group("CellData")
                _ = level_group.create_group("FieldData")

                for field_name, field_data in data[key]["CellData"].items():
                    cell_data.create_dataset(field_name, data=field_data)

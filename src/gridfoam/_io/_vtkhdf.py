from collections import defaultdict

import h5py as h5
import numpy as np
import torch

from gridfoam._base import TensorGrid
from gridfoam.cubion import NodeType, PyOctreeNode
from gridfoam.utils.enums import GridMode


def amrbox(cube: PyOctreeNode, depth: int, cell_width: int) -> np.ndarray:
    """
    Calculate the AMR box for a given octree node and depth.

    Parameters
    ----------
    cube : PyOctreeNode
        The octree node to calculate the AMR box for.
    depth : int
        The depth of the octree node.
    cell_width : int
        The width of the cell.

    Returns
    -------
    np.ndarray
        AMR box coordinates in format
        [x_start, x_end, y_start, y_end, z_start, z_end].
    """
    base = cube.cubecode.to_global_index(depth)
    start = base * cell_width
    end = start + cell_width - 1
    stacked = np.stack([start, end], axis=1)
    return stacked.reshape(-1, 6)


def save_grid(tensor_grid: TensorGrid, name: str) -> None:
    """
    Save the grid as a VTK HDF file.

    Parameters
    ----------
    tensor_grid : TensorGrid
        TensorGrid object to be saved.
    """
    output_file = tensor_grid.config.io.output_dir / name
    mode = tensor_grid.config.io.mode
    only_leaves = tensor_grid.config.io.only_leaves
    overwrite_file = tensor_grid.config.io.overwrite_file

    if output_file.exists() and not overwrite_file:
        raise FileExistsError(f"File {output_file} already exists")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    domain_width = tensor_grid.domain_width.cpu().numpy()
    cell_width = tensor_grid.config.cube.interior_width if mode == GridMode.CELL else 1
    data_width = cell_width**3

    with h5.File(output_file, "w") as f:
        # write grid description
        vtk_group = f.create_group("VTKHDF", track_order=True)
        descriptionAsASCII = "XYZ".encode("ascii")
        vtk_group.attrs.create(
            "GridDescription",
            descriptionAsASCII,
            dtype=h5.string_dtype("ascii", len(descriptionAsASCII)),
        )
        typeAsASCII = "OverlappingAMR".encode("ascii")
        vtk_group.attrs.create(
            "Type",
            typeAsASCII,
            dtype=h5.string_dtype("ascii", len(typeAsASCII)),
        )
        vtk_group.attrs["Origin"] = tensor_grid.data.domain.lower
        vtk_group.attrs["Version"] = [2, 3]

        # write grid levels
        for octree_level in tensor_grid.data.octree_levels:
            depth = octree_level.depth
            bounds = octree_level.bounds * cell_width
            dx = domain_width / bounds
            level_group = vtk_group.create_group(f"Level{depth}")
            level_group.attrs["Spacing"] = dx

            amrboxes = []
            cube_types = []
            depths = []
            field_data_by_name = defaultdict(list)
            for cube in octree_level.nodes.values():
                if only_leaves and cube.node_type != NodeType.LEAF:
                    continue
                amrboxes.append(amrbox(cube, depth, cell_width))
                cube_types.append(int(cube.node_type))
                depths.append(depth)
                if mode == GridMode.CUBE:
                    continue
                for name, field_tensor in cube.old.cells.items():
                    data = field_tensor.interior
                    extra_shape = data.shape[:-3]
                    reshaped = data.reshape(*extra_shape, -1).permute(-1, *range(len(extra_shape)))
                    field_data_by_name[name].append(reshaped)
            amrboxes = np.concatenate(amrboxes, axis=0)
            cube_types = np.array(cube_types).repeat(data_width)
            depths = np.array(depths).repeat(data_width)
            level_group.create_dataset("AMRBox", data=amrboxes)
            level_group.create_group("PointData")
            level_group.create_group("FieldData")

            celldata_group = level_group.create_group("CellData")
            celldata_group.create_dataset("cube_type", data=cube_types)
            celldata_group.create_dataset("depth", data=depths)
            if mode == GridMode.CUBE:
                continue
            for name, data_list in field_data_by_name.items():
                concatenated_data = torch.cat(data_list, dim=0).cpu().numpy()
                celldata_group.create_dataset(name, data=concatenated_data)


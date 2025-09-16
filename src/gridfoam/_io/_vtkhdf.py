from collections import defaultdict

import h5py as h5
import numpy as np
import torch

from gridfoam._base import TensorGrid
from gridfoam.cubion import NodeType, PyOctreeNode
from gridfoam.utils.enums import GridMode


def save_grid(tensor_grid: TensorGrid) -> None:
    """
    Save the grid as a VTK HDF file.

    Parameters
    ----------
    tensor_grid : TensorGrid
        TensorGrid object to be saved.
    """
    output_file = tensor_grid.config.io.output_dir / "grid.vtkhdf"
    mode = tensor_grid.config.io.mode
    only_leaves = tensor_grid.config.io.only_leaves
    overwrite_file = tensor_grid.config.io.overwrite_file

    if output_file.exists() and not overwrite_file:
        raise FileExistsError(f"File {output_file} already exists")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    domain_width = tensor_grid.domain_width.cpu().numpy()
    cell_width = tensor_grid.config.cube.width if mode == GridMode.CELL else 1

    def amrbox(octree_node: PyOctreeNode, depth: int) -> np.ndarray:
        """
        Calculate the AMR box for a given octree node and depth.

        Parameters
        ----------
        octree_node : PyOctreeNode
            The octree node to calculate the AMR box for.
        depth : int
            The depth of the octree node.

        Returns
        -------
        np.ndarray
            AMR box coordinates in format
            [x_start, x_end, y_start, y_end, z_start, z_end].
        """
        base = octree_node.cubecode.to_global_index(depth)
        match mode:
            case GridMode.CELL:
                start = base * tensor_grid.config.cube.width
                end = start + tensor_grid.config.cube.width - 1
            case GridMode.CUBE:
                start = base
                end = base
        stacked = np.stack([start, end], axis=1)
        return stacked.reshape(-1, 6)

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
            for node in octree_level.nodes.values():
                if only_leaves and node.node_type != NodeType.LEAF:
                    continue
                amrboxes.append(amrbox(node, depth))
                cube_types.append(int(node.node_type))
                depths.append(depth)
                if mode == GridMode.CUBE:
                    continue
                for name, field_tensor in node.field_tensors.items():
                    _, _, _, *extra_shape = field_tensor.interior.shape
                    data = field_tensor.interior.reshape(-1, *extra_shape)
                    field_data_by_name[name].append(data)
            amrboxes = np.concatenate(amrboxes, axis=0)
            cube_types = np.array(cube_types)
            depths = np.array(depths)
            if mode == GridMode.CELL:
                repeat = tensor_grid.config.cube.width**3
                cube_types = cube_types.repeat(repeat)
                depths = depths.repeat(repeat)
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


from collections import defaultdict
from dataclasses import dataclass

import h5py as h5
import numpy as np

from gridfoam.DNA._grid._grid import PyOctreeNode
from gridfoam.RNA.grid_handle import GridHandle
from gridfoam.RNA.registry import SimulationMetaRegistry


def _amrbox(cube: PyOctreeNode, depth: int, cell_width: int) -> np.ndarray:
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


@dataclass
class SimulationContext:
    registry: SimulationMetaRegistry
    grid_handle: GridHandle

    def save(self, filename: str) -> None:
        config = self.grid_handle.config
        output_dir = config.io.output_dir
        overwrite_file = config.io.overwrite_file
        output_file = output_dir / filename
        if output_file.exists() and not overwrite_file:
            raise FileExistsError(f"File {output_file} already exists")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        N = config.cube.interior_width
        grid = self.grid_handle.grid
        domain_width = grid.domain.upper - grid.domain.lower
        data_width = N**3

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
            vtk_group.attrs["Origin"] = grid.domain.lower
            vtk_group.attrs["Version"] = [2, 3]

            # write grid levels
            vtk_level = 0
            for octree_level in grid.octree_levels:
                depth = octree_level.depth
                bounds = octree_level.bounds * N

                amrboxes = []
                cube_types = []
                depths = []
                field_data_by_name = defaultdict(list)
                for cube in octree_level.nodes.values():
                    amrboxes.append(_amrbox(cube, depth, N))
                    cube_types.append(int(cube.node_type))
                    depths.append(depth)
                    for name, cell_field in cube.field.cells.items():
                        if not self.registry.get_field(name).default_output:
                            continue
                        C = cell_field.C
                        data = (
                            (cell_field.interior[0].reshape(C, -1))
                            .permute(1, 0)
                            .cpu()
                            .numpy()
                        )  # (N^3, C)
                        field_data_by_name[name].append(data)
                if len(amrboxes) == 0:
                    continue

                dx = domain_width / bounds
                level_group = vtk_group.create_group(f"Level{vtk_level}")
                level_group.attrs["Spacing"] = dx
                amrboxes = np.concatenate(amrboxes, axis=0)
                cube_types = np.array(cube_types).repeat(data_width)
                depths = np.array(depths).repeat(data_width)
                level_group.create_dataset("AMRBox", data=amrboxes)
                level_group.create_group("PointData")
                level_group.create_group("FieldData")

                celldata_group = level_group.create_group("CellData")
                celldata_group.create_dataset("cube_type", data=cube_types)
                celldata_group.create_dataset("depth", data=depths)
                vtk_level += 1

                # write field data
                for name, data_list in field_data_by_name.items():
                    concatenated_data = np.concatenate(data_list, axis=0)
                    celldata_group.create_dataset(name, data=concatenated_data)

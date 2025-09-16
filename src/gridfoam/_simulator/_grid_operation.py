import numpy as np
import torch
from jaxtyping import Int

from gridfoam._base import TensorGrid
from gridfoam.cubion import PyCubeCode


def generate_grid_indices(
    divisions: Int[torch.Tensor, " 3"],
) -> Int[torch.Tensor, "n_grid 3"]:
    """
    Generate 3D grid indices
    according to the number of divisions along each axis.

    The output indices are ordered in Z-order.

    Parameters
    ----------
    divisions : Int32[torch.Tensor, " 3"]
        The number of divisions along each axis (X, Y, Z).

    Returns
    -------
    Int32[torch.Tensor, "n_grid 3"]
        Tensor of shape (n_grid, 3) containing all grid indices,
        where n_grid = divisions[0] * divisions[1] * divisions[2].
        Each row corresponds to a (ix, iy, iz) in the grid.
    """
    indices_per_axis = [
        torch.arange(n, dtype=divisions.dtype, device=divisions.device)
        for n in reversed(divisions)
    ]
    indices = torch.meshgrid(*indices_per_axis, indexing="ij")
    indices = torch.stack(indices[::-1], dim=-1).reshape(-1, 3)
    return indices


def init_p(grid: TensorGrid) -> None:
    """Initialize pressure field with radial distance from origin.

    This method initializes the pressure field for all cubes by calculating
    the radial distance from the origin for each cell center. The pressure
    is set to the distance from the origin (sqrt(x^2 + y^2 + z^2)).

    After initialization, ghost cells are synchronized from children and
    halo cells are updated.
    """
    cell_width = grid.config.cube.width
    for octree_level in grid.data.octree_levels:
        depth = octree_level.depth
        bounds = octree_level.bounds * cell_width
        dx = grid.domain_width / bounds
        half_dx = dx * 0.5
        for cube in octree_level.nodes.values():
            # if cube.cube_type != CubeType.LEAF:
            #     continue
            cubecode: PyCubeCode = cube.cubecode
            width = grid.config.cube.width
            divisions = torch.tensor(
                [width] * 3, dtype=torch.int64, device=grid.device
            )
            global_index = torch.tensor(
                cubecode.to_global_index(depth).astype(np.int64),
                dtype=torch.int64,
                device=grid.device,
            )
            local_cell_indices = generate_grid_indices(divisions).reshape(
                width, width, width, 3
            )
            global_cell_indices = (
                global_index[None, None, None, :] * width + local_cell_indices
            )
            global_lower_position = torch.tensor(
                grid.data.domain.lower, device=grid.device, dtype=torch.float32
            )
            global_cell_position = (
                global_lower_position
                + (
                    2.0
                    * global_cell_indices.to(dtype=global_lower_position.dtype)[
                        ..., :
                    ]
                    + 1
                )
                * half_dx
            )
            x = global_cell_position[..., 0]
            y = global_cell_position[..., 1]
            z = global_cell_position[..., 2]
            r = torch.sqrt(x**2 + y**2 + z**2)
            # r = (z-self.domain.center[2])**2
            # r = x+y+z
            cube.field_tensors["p"].interior = r[..., None]
    # propagate to ghost cubes
    # self.sync_ghost_from_children()
    # self.sync_ghost_from_parent()

    # update halo cells
    grid.update_halo()


def grad_p(grid: TensorGrid) -> None:
    """Calculate pressure gradient using finite difference method.

    This method computes the pressure gradient for all cubes using first-order
    finite differences. The gradient is calculated in x, y, and z directions
    using the pressure values at neighboring cells.

    After calculation, ghost cells are synchronized from children and
    halo cells are updated.
    """
    cell_width = grid.config.cube.width
    for octree_level in grid.data.octree_levels:
        bounds = octree_level.bounds * cell_width
        dx = grid.domain_width / bounds
        for cube in octree_level.nodes.values():
            # if cube.cube_type != CubeType.LEAF:
            #     continue
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
            dpdx = (p_i_xp - p_i) / dx[0]
            dpdy = (p_i_yp - p_i) / dx[1]
            dpdz = (p_i_zp - p_i) / dx[2]
            cube.field_tensors["grad_p"].interior = torch.cat(
                [dpdx, dpdy, dpdz], dim=-1
            )
    # propagate to ghost cubes
    # self.sync_ghost_from_children()
    # self.sync_ghost_from_parent()

    # update halo cells
    grid.update_halo()

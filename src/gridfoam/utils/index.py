import torch
from jaxtyping import Int32


def get_grid_indices(
    divisions: Int32[torch.Tensor, " 3"],
) -> Int32[torch.Tensor, "n_grid 3"]:
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
        torch.arange(n, dtype=torch.int32) for n in reversed(divisions)
    ]
    indices = torch.meshgrid(*indices_per_axis, indexing="ij")
    indices = torch.stack(indices[::-1], dim=-1).reshape(-1, 3)
    return indices


def linearize_grid_indices(
    indices: Int32[torch.Tensor, "n_grid 3"],
    divisions: Int32[torch.Tensor, " 3"],
) -> list[int]:
    """
    Convert 3D grid indices to 1D linearized indices.
    [..., [ix, iy, iz], ...]
    ->
    [..., (ix + (iy + iz * divisions[1]) * divisions[0]), ...]

    Parameters
    ----------
    indices : Int32[torch.Tensor, "n_grid 3"]
        Tensor of shape (n_grid, 3) containing 3D grid indices (ix, iy, iz).
    divisions : Int32[torch.Tensor, " 3"]
        Number of divisions along each axis (X, Y, Z).

    Returns
    -------
    list[int]
        List of grid index
        The length of the list is n_grid, where
        n_grid = divisions[0] * divisions[1] * divisions[2].
    """
    XX, YY, ZZ = indices.T
    XX, YY, ZZ = XX.to(torch.int64), YY.to(torch.int64), ZZ.to(torch.int64)
    div_x, div_y = (
        divisions[0].to(torch.int64),
        divisions[1].to(torch.int64),
    )
    index = XX + (YY + ZZ * div_y) * div_x
    return index.tolist()

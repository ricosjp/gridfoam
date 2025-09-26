import torch
from jaxtyping import Int


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
        The number of divisions along each axis (Z, Y, X).

    Returns
    -------
    Int32[torch.Tensor, "n_grid 3"]
        Tensor of shape (n_grid, 3) containing all grid indices,
        where n_grid = divisions[0] * divisions[1] * divisions[2].
        Each row corresponds to a (iz, iy, ix) in the grid.
    """
    indices_per_axis = [
        torch.arange(n, dtype=divisions.dtype, device=divisions.device)
        for n in reversed(divisions)
    ]
    indices = torch.meshgrid(*indices_per_axis, indexing="ij")
    indices = torch.stack(indices, dim=-1).reshape(-1, 3)
    return indices

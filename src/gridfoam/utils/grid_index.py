import torch
from jaxtyping import Int


def generate_grid_indices(
    divisions: Int[torch.Tensor, " 3"],
) -> Int[torch.Tensor, "3 n_grid"]:
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
    Int32[torch.Tensor, "3 n_grid"]
        Tensor of shape (3, n_grid) containing all grid indices,
        where n_grid = divisions[0] * divisions[1] * divisions[2].
        Each row corresponds:
        - 0: x-axis
        - 1: y-axis
        - 2: z-axis
    """
    indices_per_axis = [
        torch.arange(n, dtype=divisions.dtype, device=divisions.device)
        for n in reversed(divisions)
    ]
    indices = torch.meshgrid(*indices_per_axis, indexing="ij")
    indices = torch.stack(indices[::-1], dim=0).reshape(3, -1)
    return indices

import torch
from jaxtyping import Int32


def generate_grid_indices(
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


def ravel_index_3d(
    indices: Int32[torch.Tensor, "... 3"],
    divisions: Int32[torch.Tensor, " 3"],
) -> Int32[torch.Tensor, " ..."]:
    """
    Convert 3D grid indices to 1D linearized indices.
    [..., [ix, iy, iz], ...]
    ->
    [..., (ix + (iy + iz * divisions[1]) * divisions[0]), ...]

    Parameters
    ----------
    indices : Int32[torch.Tensor, "... 3"]
        Tensor of shape (..., 3) containing 3D grid indices (ix, iy, iz).
    divisions : Int32[torch.Tensor, " 3"]
        Number of divisions along each axis (X, Y, Z).

    Returns
    -------
    Int32[torch.Tensor, "..."]
        Tensor of shape (...,) containing linearized grid indices.
    """
    ix = indices[..., 0]
    iy = indices[..., 1]
    iz = indices[..., 2]
    index = ix + (iy + iz * divisions[1]) * divisions[0]
    return index

def unravel_index_3d(
    index: Int32[torch.Tensor, " ..."],
    divisions: Int32[torch.Tensor, " 3"],
) -> Int32[torch.Tensor, "... 3"]:
    """
    Convert 1D linearized indices to 3D grid indices.
    [..., (ix + (iy + iz * divisions[1]) * divisions[0]), ...]
    ->
    [..., [ix, iy, iz], ...]

    Parameters
    ----------
    index : Int32[torch.Tensor, "..."]
        Tensor of shape (...,) containing linearized grid indices.
    divisions : Int32[torch.Tensor, " 3"]
        Number of divisions along each axis (X, Y, Z).

    Returns
    -------
    Int32[torch.Tensor, "... 3"]
        Tensor of shape (..., 3) containing 3D grid indices (ix, iy, iz).
    """
    iz = index // (divisions[0] * divisions[1])
    iy = (index % (divisions[0] * divisions[1])) // divisions[0]
    ix = index % divisions[0]
    return torch.stack([ix, iy, iz], dim=-1)

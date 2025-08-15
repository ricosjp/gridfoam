from itertools import product

import torch
from jaxtyping import Int32

from gridfoam.utils.enums import AddressMode


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
    linearized_index: Int32[torch.Tensor, " ..."] | int,
    divisions: Int32[torch.Tensor, " 3"],
) -> Int32[torch.Tensor, "... 3"]:
    """
    Convert 1D linearized indices to 3D grid indices.
    [..., (ix + (iy + iz * divisions[1]) * divisions[0]), ...]
    ->
    [..., [ix, iy, iz], ...]

    Parameters
    ----------
    linearized_index : Int32[torch.Tensor, "..."] | int
        Linearized grid indices.
    divisions : Int32[torch.Tensor, " 3"]
        Number of divisions along each axis (X, Y, Z).

    Returns
    -------
    Int32[torch.Tensor, "... 3"]
        Tensor of shape (..., 3) containing 3D grid indices (ix, iy, iz).
    """
    if isinstance(linearized_index, int):
        linearized_index = torch.tensor(linearized_index, dtype=torch.int32)
    iz = linearized_index // (divisions[0] * divisions[1])
    iy = (linearized_index % (divisions[0] * divisions[1])) // divisions[0]
    ix = linearized_index % divisions[0]
    return torch.stack([ix, iy, iz], dim=-1)


def neighbor_indices(
    indices: Int32[torch.Tensor, "... 3"],
    divisions: Int32[torch.Tensor, " 3"],
    include_self: bool = False,
    address_mode: AddressMode = AddressMode.CLAMP,
) -> Int32[torch.Tensor, "... n_neighbors 3"]:
    """
    Get the 26-neighborhood indices for a given set of indices.
    If neighbor indices are outside the domain,
    their values are handled according to `address_mode`.

    Let k be an index outside the valid range:
    - for `WRAP`, return k % n
    - for `CLAMP`, return 0 for k < 0 and n-1 for k >= n
    - for `BORDER`, return -1 for k < 0 or k >= n

    Parameters
    ----------
    indices : Int32[torch.Tensor, "... 3"]
        Tensor of shape (..., 3) containing the indices.
    divisions : Int32[torch.Tensor, " 3"]
        Number of divisions along each axis (X, Y, Z).
    include_self : bool, default=False
        Whether to include the center index itself as a neighbor.
    address_mode : AddressMode, default=AddressMode.CLAMP
        Addressing mode for out-of-domain neighbors.

    Returns
    -------
    Int32[torch.Tensor, "... n_neighbors 3"]
        Tensor of shape (..., n_neighbors, 3)
        containing the 26-neighborhood indices.
    """
    offsets = torch.tensor(
        list(product([-1, 0, 1], repeat=3)), dtype=torch.int32
    )
    if not include_self:
        offsets = offsets[(offsets != 0).any(dim=1)]
    offsets = offsets[:, [2, 1, 0]]

    # (..., 3) + (26, 3) -> (..., 26, 3)
    neighbors = indices[..., None, :] + offsets

    match address_mode:
        case AddressMode.WRAP:
            neighbors = neighbors % divisions
        case AddressMode.CLAMP:
            neighbors = torch.clamp(
                neighbors, torch.zeros_like(divisions), divisions - 1
            )
        case AddressMode.BORDER:
            is_inside = ((neighbors >= 0) & (neighbors < divisions)).all(dim=-1)
            neighbors[~is_inside] = -1
    return neighbors

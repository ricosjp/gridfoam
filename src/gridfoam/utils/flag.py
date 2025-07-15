from itertools import product

import torch
from jaxtyping import Int32


def dilate_sparse_coords(
    flag_coords: Int32[torch.Tensor, "n_flag 3"],
    bounds: Int32[torch.Tensor, " 3"],
) -> Int32[torch.Tensor, "n_flag 3"]:
    """
    Dilate sparse flag coordinates (ix, iy, iz) to their 26-neighborhood.

    Parameters
    ----------
    flag_coords : Int32[torch.Tensor, "n_flag 3"]
        Tensor of shape (n_flag, 3) containing the coordinates to be dilated.
    bounds : Int32[torch.Tensor, " 3"]
        Tensor of shape (3,) specifying the spatial bounds.

    Returns
    -------
    Int32[torch.Tensor, "n_flag_dilated 3"]
        Tensor of shape (n_flag_dilated, 3) containing the dilated coordinates.
    """
    D, H, W = bounds

    # 27-neighborhood offsets
    offsets = torch.tensor(
        list(product([-1, 0, 1], repeat=3)), dtype=torch.int32
    )

    # (N, 1, 3) + (27, 3) -> (N, 27, 3)
    expanded = flag_coords[:, None, :] + offsets[None, :, :]
    all_coords = expanded.reshape(-1, 3)

    # Remove duplicates
    dilated = torch.unique(all_coords, dim=0)

    # Remove coordinates outside the bounds
    mask = (
        (dilated[:, 0] >= 0)
        & (dilated[:, 0] < D)
        & (dilated[:, 1] >= 0)
        & (dilated[:, 1] < H)
        & (dilated[:, 2] >= 0)
        & (dilated[:, 2] < W)
    )
    return dilated[mask]

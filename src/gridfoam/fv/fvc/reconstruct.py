"""Public velocity reconstruction from volumetric face flux."""

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid


def _accumulate_reconstruct(
    numer: Float[torch.Tensor, " C 3"],
    denom: Float[torch.Tensor, " C 3"],
    cells: torch.Tensor,
    Sf: Float[torch.Tensor, " F 3"],
    flux: Float[torch.Tensor, " F 1"],
) -> None:
    """
    Accumulate ``n_hat * phi`` and diagonal ``|Sf| n n^T`` for ``cells``.
    """
    mag_Sf = torch.linalg.vector_norm(Sf, dim=1, keepdim=True)
    n_hat = Sf / mag_Sf
    numer.index_add_(0, cells, n_hat * flux)
    # Diagonal of |Sf| n n^T; exact for axis-aligned faces.
    denom.index_add_(0, cells, mag_Sf * n_hat * n_hat)


def reconstruct(
    phi: FaceField,
    U: CellField,
) -> Float[torch.Tensor, " C 3"]:
    """
    Reconstruct cell-centered velocity from volumetric face flux.

    Implements OpenFOAM ``fvc::reconstruct``:

    .. math::

        \\mathbf{U}_P = \\left( \\sum_f |S_f| \\hat{n}_f \\hat{n}_f^T
        \\right)^{-1} \\sum_f \\hat{n}_f \\phi_f

    where the sums run over every face of the cell (``surfaceSum``), so
    owner and neighbour receive the same-signed contribution. On an
    axis-aligned octree the reference tensor is diagonal, so each velocity
    component is the area-weighted average of the face-normal velocities
    on that axis. Hanging-node sub-faces are handled naturally because both
    sums are area weighted.

    Parameters
    ----------
    phi : FaceField
        Scalar volumetric face flux field (``U & Sf``).
    U : CellField
        Target velocity field that receives the reconstructed data.

    Returns
    -------
    torch.Tensor
        Reconstructed velocity data with shape ``[num_cells, 3]``.
    """
    if phi.num_components != 1:
        raise ValueError("phi must be a scalar face flux field.")
    if U.num_components != 3:
        raise ValueError("U must be a 3-component velocity field.")
    if phi.grid is not U.grid:
        raise ValueError("phi and U must share the same grid.")

    grid = phi.grid
    numer = torch.zeros(
        (grid.num_cells, 3), dtype=grid.dtype, device=grid.device
    )
    denom = torch.zeros_like(numer)

    single_mask = phi.single_mask
    Sf_single = grid.Sf[single_mask]
    _accumulate_reconstruct(
        numer, denom, grid.owner[single_mask], Sf_single, phi.single_data
    )
    _accumulate_reconstruct(
        numer, denom, grid.neighbour[single_mask], Sf_single, phi.single_data
    )
    _accumulate_reconstruct(
        numer,
        denom,
        grid.domain_bnd_owner,
        grid.domain_bnd_Sf,
        phi.domain_bnd_data,
    )

    if isinstance(grid, AxisProjectedGrid) and grid.num_immersed_faces > 0:
        immersed = grid.ap_is_immersed_faces
        immersed_Sf = grid.Sf[immersed]
        _accumulate_reconstruct(
            numer, denom, grid.owner[immersed], immersed_Sf, phi.immersed_upper
        )
        _accumulate_reconstruct(
            numer,
            denom,
            grid.neighbour[immersed],
            -immersed_Sf,
            phi.immersed_lower,
        )

    u_data = numer / denom
    U.data = u_data
    return u_data

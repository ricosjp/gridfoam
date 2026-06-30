import torch
from jaxtyping import Float

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv import fvc
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)


def correct_flux(
    phi: FaceField,
    U: CellField,
    update_internal: bool = False,
):
    """
    Synchronize face flux ``phi`` from velocity field ``U``.

    Parameters
    ----------
    phi : FaceField
        Target face-flux field to update.
    U : CellField
        Velocity field used to reconstruct fluxes.
    update_internal : bool, optional
        If ``True``, internal-face fluxes are recomputed
        from interpolated ``U``.
        If ``False``, only boundary-face fluxes are synchronized from BC states.
    """
    grid = U.grid

    # Internal face flux initialization/synchronization
    if update_internal:
        U_f = fvc.interpolate(U)
        # Flux from inner product with face area vector Sf.
        phi.single_data = torch.sum(
            U_f.single_data * grid.Sf[U_f.single_mask], dim=1, keepdim=True
        )
        phi.domain_bnd_data = torch.sum(
            U_f.domain_bnd_data * grid.domain_bnd_Sf, dim=1, keepdim=True
        )
        if isinstance(grid, AxisProjectedGrid):
            phi.immersed_upper = torch.sum(
                U_f.immersed_upper * grid.Sf[grid.ap_is_immersed_faces],
                dim=1,
                keepdim=True,
            )
            phi.immersed_lower = torch.sum(
                U_f.immersed_lower * grid.Sf[grid.ap_is_immersed_faces],
                dim=1,
                keepdim=True,
            )

    for batch in iter_boundary_batches(U):
        _, _, _, U_b = evaluate_boundary_state(U, batch)
        # Domain boundary flux synchronization
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            Sf_bnd = grid.domain_bnd_Sf[batch.face_mask]
            phi.domain_bnd_data[batch.face_mask] = torch.sum(
                U_b * Sf_bnd, dim=1, keepdim=True
            )
            continue

        # Immersed boundary flux synchronization
        if isinstance(grid, AxisProjectedGrid):
            immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
            if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
                Sf_upper = immersed_Sf[batch.face_mask]
                phi.immersed_upper[batch.face_mask] = torch.sum(
                    U_b * Sf_upper, dim=1, keepdim=True
                )
                continue
            elif batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                Sf_lower = -immersed_Sf[batch.face_mask]
                phi.immersed_lower[batch.face_mask] = torch.sum(
                    U_b * Sf_lower, dim=1, keepdim=True
                )
                continue


def reconstruct_U_from_phi(
    phi: FaceField,
    U: CellField,
) -> Float[torch.Tensor, " C 3"]:
    """
    Reconstruct cell-centered velocity from volumetric face flux.

    Implements OpenFOAM ``fvc::reconstruct``:

    .. math::

        \\mathbf{U}_P = \\frac{1}{V_P}
        \\sum_f (\\text{oriented } \\phi_f \\mathbf{S}_f)

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
    u_data = torch.zeros(
        (grid.num_cells, 3), dtype=grid.dtype, device=grid.device
    )

    single_mask = phi.single_mask
    flux_vec = phi.single_data * grid.Sf[single_mask]
    u_data.index_add_(0, grid.owner[single_mask], flux_vec)
    u_data.index_add_(0, grid.neighbour[single_mask], -flux_vec)

    flux_domain = phi.domain_bnd_data * grid.domain_bnd_Sf
    u_data.index_add_(0, grid.domain_bnd_owner, flux_domain)

    if isinstance(grid, AxisProjectedGrid):
        immersed_owner = grid.owner[grid.ap_is_immersed_faces]
        immersed_neighbour = grid.neighbour[grid.ap_is_immersed_faces]
        immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
        u_data.index_add_(0, immersed_owner, phi.immersed_upper * immersed_Sf)
        u_data.index_add_(
            0, immersed_neighbour, -phi.immersed_lower * immersed_Sf
        )

    u_data = u_data / grid.cell_volumes
    U.data = u_data
    return u_data

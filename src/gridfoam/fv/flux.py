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


def set_phi_from_matrix_flux(
    phi: FaceField,
    flux_data: Float[torch.Tensor, " F_single k"],
) -> None:
    """
    Write matrix face flux values to single-sided internal faces.

    Parameters
    ----------
    phi : FaceField
        Target volumetric face-flux field.
    flux_data : torch.Tensor
        Flux values with shape ``[F_single, k]``.
    """
    phi.single_data = flux_data


def compute_phi_hbya(
    phi: FaceField, HbyA: CellField
) -> Float[torch.Tensor, " F_single 1"]:
    """
    Compute ``phi_HbyA = HbyA_f & Sf`` on single-sided internal faces.

    Domain and immersed boundary fluxes are left to ``correct_flux`` /
    ``finalize_pressure_correction``.

    Parameters
    ----------
    phi : FaceField
        Face flux field whose ``single_data`` is updated.
    HbyA : CellField
        Momentum predictor ``H(U) / A(U)`` field.

    Returns
    -------
    torch.Tensor
        Computed ``phi_HbyA`` with shape ``[F_single, 1]``.
    """
    grid = HbyA.grid
    HbyA_f = fvc.interpolate(HbyA)
    phi_hbya = torch.sum(
        HbyA_f.single_data * grid.Sf[HbyA_f.single_mask],
        dim=1,
        keepdim=True,
    )
    phi.single_data = phi_hbya
    return phi_hbya


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

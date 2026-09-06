"""
Face-flux helpers shared by the pressure-velocity algorithms.

Flux orientation
----------------
``phi.single_data`` is oriented along ``Sf`` (owner to neighbour).
Boundary blocks are oriented **outward from their adjacent cell**:
``domain_bnd_data`` along ``domain_bnd_Sf``, ``immersed_upper`` along
``+Sf`` (outward from the owner) and ``immersed_lower`` along ``-Sf``
(outward from the neighbour). A positive boundary flux is therefore always
outflow, which is what the boundary treatment in ``fvm.div``,
``inletOutlet`` and ``fixedFluxPressure`` assume.
"""

from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv import fvc
from gridfoam.fv.boundary_ops import (
    boundary_block,
    iter_boundary_states,
    outward_boundary_Sf,
    uncovered_domain_faces,
)
from gridfoam.fv.kernels.face_geometry import face_geometry


def flux_from_face_velocity(
    phi: FaceField, U_f: FaceField, U: CellField
) -> None:
    """
    Fill every block of ``phi`` with ``U_f & Sf`` using outward normals.

    Domain patches on which ``U`` has no boundary condition (``empty``
    patches of 2-D cases) carry no flux: the extrapolated face velocity is
    ignored there so that the boundary flux sum stays consistent with the
    patches actually seen by ``fvm.div`` and ``adjust_phi``.

    Parameters
    ----------
    phi : FaceField
        Target flux field.
    U_f : FaceField
        Face velocity (e.g. ``fvc.interpolate(U)`` or of ``HbyA``).
    U : CellField
        Velocity field whose boundary conditions define the flux patches.
    """
    grid = phi.grid
    geo = face_geometry(grid)
    phi.single_data = torch.sum(U_f.single_data * geo.Sf_s, dim=1, keepdim=True)
    domain_flux = torch.sum(
        U_f.domain_bnd_data * grid.domain_bnd_Sf, dim=1, keepdim=True
    )
    empty = uncovered_domain_faces(U)
    phi.domain_bnd_data = torch.where(
        empty[:, None], torch.zeros_like(domain_flux), domain_flux
    )
    if isinstance(grid, AxisProjectedGrid) and grid.num_immersed_faces > 0:
        immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
        phi.immersed_upper = torch.sum(
            U_f.immersed_upper * immersed_Sf, dim=1, keepdim=True
        )
        phi.immersed_lower = torch.sum(
            U_f.immersed_lower * (-immersed_Sf), dim=1, keepdim=True
        )


def _constrain_hbya_boundary_flux(phi_hbya: FaceField, U: CellField) -> None:
    """
    Impose ``U_b & Sf`` on non-assignable velocity patches (OpenFOAM
    ``constrainHbyA``).
    """
    grid = U.grid
    for state in iter_boundary_states(U):
        batch = state.batch
        if U.bcs[batch.patch_name].assignable:
            continue
        Sf_out = outward_boundary_Sf(grid, batch)
        boundary_block(phi_hbya, batch.face_kind)[batch.face_mask] = torch.sum(
            state.psi_b * Sf_out, dim=1, keepdim=True
        )


def compute_phi_hbya(
    phi_hbya: FaceField,
    HbyA: CellField,
    U: CellField,
    *,
    ddt_corr: Float[torch.Tensor, " F_single 1"] | None = None,
) -> FaceField:
    """
    Build the predicted flux ``phiHbyA = flux(constrainHbyA(HbyA, U))``.

    Internal faces use the interpolated ``HbyA``; boundary faces use the
    zero-gradient extrapolated ``HbyA`` except on velocity patches whose
    boundary condition is not ``assignable`` (fixed value, slip, immersed
    walls), where the velocity boundary flux ``U_b & Sf`` is imposed as in
    OpenFOAM ``constrainHbyA``. An optional ``ddtCorr`` contribution is
    added on single-sided internal faces.

    Parameters
    ----------
    phi_hbya : FaceField
        Target field receiving the predicted flux on all blocks.
    HbyA : CellField
        Momentum predictor ``H(U) / A(U)`` (possibly SIMPLEC-adjusted).
    U : CellField
        Velocity field supplying the boundary conditions.
    ddt_corr : torch.Tensor or None, optional
        Pre-multiplied time-derivative flux correction
        ``interpolate(rAU) * ddtCorr(U, phi)`` on single-sided faces.

    Returns
    -------
    FaceField
        ``phi_hbya`` after the update.
    """
    HbyA_f = fvc.interpolate(HbyA)
    flux_from_face_velocity(phi_hbya, HbyA_f, U)
    if ddt_corr is not None:
        phi_hbya.single_data = phi_hbya.single_data + ddt_corr
    _constrain_hbya_boundary_flux(phi_hbya, U)
    return phi_hbya


def correct_flux(
    phi: FaceField,
    U: CellField,
    update_internal: bool = False,
) -> None:
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
    if update_internal:
        flux_from_face_velocity(phi, fvc.interpolate(U), U)

    grid = U.grid
    for state in iter_boundary_states(U):
        batch = state.batch
        Sf_out = outward_boundary_Sf(grid, batch)
        boundary_block(phi, batch.face_kind)[batch.face_mask] = torch.sum(
            state.psi_b * Sf_out, dim=1, keepdim=True
        )

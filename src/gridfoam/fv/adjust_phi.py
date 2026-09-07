"""OpenFOAM ``adjustPhi`` equivalent for domain boundary flux scaling."""

from __future__ import annotations

import torch

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.derived.inlet_outlet import InletOutletBC
from gridfoam.core.field import CellField, FaceField
from gridfoam.meta.enums import DomainBoundaryPatch
from gridfoam.meta.types import PatchName


def _patch_fixes_value(U: CellField, patch: PatchName) -> bool:
    """
    Return whether the velocity BC on a patch fixes the value.

    Parameters
    ----------
    U : CellField
        Velocity field whose boundary conditions are inspected.
    patch : PatchName
        Domain-boundary patch name.

    Returns
    -------
    bool
        ``True`` for fixed-value patches. ``False`` for adjustable patches
        such as ``inletOutlet``, slip, and Neumann boundaries.

    Notes
    -----
    OpenFOAM treats ``inletOutlet`` patches as adjustable even when they
    apply Dirichlet values on inflow faces.
    """
    bc = U.bcs.get(patch)
    if bc is None:
        return False
    if isinstance(bc, InletOutletBC):
        return False
    return isinstance(bc, DirichletBC)


def adjust_phi(
    phi: FaceField, U: CellField, p: CellField | None = None
) -> bool:
    """
    Scale adjustable outlet boundary fluxes to satisfy global continuity.

    Implements OpenFOAM ``adjustPhi``. Inflow (``phi < 0``) is accumulated
    and balanced against fixed-value and adjustable outflow patches:

    ``massCorr = (massIn - fixedMassOut) / adjustableMassOut``.

    Only outflow faces (``phi > 0``) on adjustable patches are scaled. The
    function is meant to be applied to the predicted flux ``phiHbyA``
    before the pressure equation is assembled, so that the pure-Neumann
    Poisson problem has a compatible right-hand side.

    Parameters
    ----------
    phi : FaceField
        Face flux field to adjust on domain boundaries.
    U : CellField
        Velocity field used to classify patch adjustability.
    p : CellField or None, optional
        Pressure field. As in OpenFOAM, no adjustment is made when the
        pressure level is fixed by a Dirichlet condition (the flux balance
        is then set by the pressure solution). ``None`` always adjusts.

    Returns
    -------
    bool
        ``True`` when the flux was adjusted.
    """
    if p is not None and any(
        isinstance(bc, DirichletBC) for bc in p.bcs.values()
    ):
        # Pressure level is fixed: OpenFOAM ``adjustPhi`` is a no-op.
        return False
    grid = phi.grid
    mass_in = torch.zeros((), dtype=grid.dtype, device=grid.device)
    fixed_mass_out = torch.zeros((), dtype=grid.dtype, device=grid.device)
    adjustable_mass_out = torch.zeros((), dtype=grid.dtype, device=grid.device)

    for patch in U.bcs:
        if not isinstance(patch, DomainBoundaryPatch):
            continue

        mask = grid.get_domain_bnd_mask(patch)
        if not torch.any(mask):
            continue

        phip = phi.domain_bnd_data[mask]
        inflow = phip < 0.0
        outflow = phip > 0.0

        mass_in = mass_in - phip[inflow].sum()
        if _patch_fixes_value(U, patch):
            fixed_mass_out = fixed_mass_out + phip[outflow].sum()
        else:
            adjustable_mass_out = adjustable_mass_out + phip[outflow].sum()

    if adjustable_mass_out.item() <= 0.0:
        return False

    mass_corr = (mass_in - fixed_mass_out) / adjustable_mass_out

    for patch in U.bcs:
        if not isinstance(patch, DomainBoundaryPatch):
            continue
        if _patch_fixes_value(U, patch):
            continue

        mask = grid.get_domain_bnd_mask(patch)
        if not torch.any(mask):
            continue

        phip = phi.domain_bnd_data[mask]
        outflow = phip > 0.0
        if torch.any(outflow):
            phi.domain_bnd_data[mask] = torch.where(
                outflow,
                phip * mass_corr,
                phip,
            )
    return True

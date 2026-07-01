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


def adjust_phi(phi: FaceField, U: CellField) -> None:
    """
    Scale adjustable outlet boundary fluxes to satisfy global continuity.

    Implements OpenFOAM ``adjustPhi``. Inflow (``phi < 0``) is accumulated
    and balanced against fixed-value and adjustable outflow patches:

    ``massCorr = (massIn - fixedMassOut) / adjustableMassOut``.

    Only outflow faces (``phi > 0``) on adjustable patches are scaled.

    Parameters
    ----------
    phi : FaceField
        Face flux field to adjust on domain boundaries.
    U : CellField
        Velocity field used to classify patch adjustability.
    """
    grid = phi.grid
    mass_in = torch.zeros((), dtype=grid.dtype, device=grid.device)
    fixed_mass_out = torch.zeros((), dtype=grid.dtype, device=grid.device)
    adjustable_mass_out = torch.zeros((), dtype=grid.dtype, device=grid.device)

    for patch, bc in U.bcs.items():
        if not isinstance(patch, DomainBoundaryPatch):
            continue
        if bc is None:
            continue

        mask = grid.get_domain_bnd_mask(patch)
        if not torch.any(mask):
            continue

        phip = phi.domain_bnd_data[mask, 0]
        inflow = phip < 0.0
        outflow = phip > 0.0

        mass_in = mass_in - phip[inflow].sum()
        if _patch_fixes_value(U, patch):
            fixed_mass_out = fixed_mass_out + phip[outflow].sum()
        else:
            adjustable_mass_out = adjustable_mass_out + phip[outflow].sum()

    if adjustable_mass_out.item() <= 0.0:
        return

    mass_corr = (mass_in - fixed_mass_out) / adjustable_mass_out

    for patch in U.bcs:
        if not isinstance(patch, DomainBoundaryPatch):
            continue
        if _patch_fixes_value(U, patch):
            continue

        mask = grid.get_domain_bnd_mask(patch)
        if not torch.any(mask):
            continue

        phip = phi.domain_bnd_data[mask, 0]
        outflow = phip > 0.0
        if torch.any(outflow):
            phi.domain_bnd_data[mask, 0] = torch.where(
                outflow,
                phip * mass_corr,
                phip,
            )

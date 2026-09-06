"""Time-derivative flux correction (OpenFOAM ``fvc::ddtCorr``)."""

from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField, FaceField
from gridfoam.fv.kernels.face_geometry import face_geometry


def ddt_corr(
    U: CellField, phi: FaceField
) -> Float[torch.Tensor, " F_single 1"]:
    """
    Euler time-derivative flux correction on single-sided internal faces.

    Implements OpenFOAM ``EulerDdtScheme::fvcDdtPhiCorr``:

    ``ddtCorr = ddtCouplingCoeff * (phi^0 - U^0_f & Sf) / dt``,

    ``ddtCouplingCoeff = 1 - min(|phi^0 - U^0_f & Sf| / (|phi^0| + eps), 1)``.

    The term re-introduces the previous time-level face flux into the
    predicted flux ``phiHbyA`` (pre-multiplied by ``interpolate(rAU)`` by
    the caller) and removes the time-step dependence of the Rhie-Chow
    coupling in PISO/PIMPLE. Boundary faces receive no correction, matching
    OpenFOAM for non-coupled patches. ``U^0_f`` uses the plain two-point
    linear interpolation.

    Parameters
    ----------
    U : CellField
        Velocity field with a stored old time level (``FieldRole.TRANSIENT``).
    phi : FaceField
        Face flux whose ``update_history`` is called once per time step.

    Returns
    -------
    torch.Tensor
        Flux correction with shape ``[F_single, 1]``.
    """
    grid = U.grid
    geo = face_geometry(grid)
    U0 = U.old_data
    U0_f = geo.w_s * U0[geo.owner_s] + (1.0 - geo.w_s) * U0[geo.neighbour_s]
    phi0 = phi.old_single_data
    phi_corr = phi0 - torch.sum(U0_f * geo.Sf_s, dim=1, keepdim=True)

    eps = torch.finfo(phi_corr.dtype).tiny
    coupling = 1.0 - torch.clamp(phi_corr.abs() / (phi0.abs() + eps), max=1.0)
    return coupling * phi_corr / grid.dt

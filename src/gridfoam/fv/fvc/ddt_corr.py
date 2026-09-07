"""Time-derivative flux correction (OpenFOAM ``fvc::ddtCorr``)."""

from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField, FaceField
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.fv.schemes.ddt import ddt_coefficients


def ddt_corr(U: CellField, phi: FaceField) -> Float[torch.Tensor, " F_single"]:
    """
    Euler/backward flux correction on single-sided internal faces.

    Implements OpenFOAM ``EulerDdtScheme::fvcDdtPhiCorr``:

    ``ddtCorr = ddtCouplingCoeff * (phi^0 - U^0_f & Sf) / dt``,

    ``ddtCouplingCoeff = 1 - min(|phi^0 - U^0_f & Sf| / (|phi^0| + eps), 1)``.

    Backward uses ``b*phiCorr_old - c*phiCorr_older`` with the same time
    weights as ``fvm.ddt``. The coupling coefficient is computed from only
    the immediately previous level, as in OpenCFD v2606.

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
        Flux correction with shape ``[F_single]``.
    """
    if U.component_shape != (3,) or phi.component_shape != ():
        raise ValueError("ddt_corr requires vector U and scalar phi")
    grid = U.grid
    geo = face_geometry(grid)
    U0 = U.old_data
    U0_f = (
        geo.w_s[:, None] * U0[geo.owner_s]
        + (1.0 - geo.w_s)[:, None] * U0[geo.neighbour_s]
    )
    phi0 = phi.old_single_data
    phi_corr = phi0 - torch.sum(U0_f * geo.Sf_s, dim=1)

    _, b, c = ddt_coefficients(U)
    history_correction = b * phi_corr
    if c != 0.0:
        if phi.older_single_data is None or phi.previous_dt != U.previous_dt:
            raise ValueError(
                "backward ddtCorr requires synchronized U and phi histories"
            )
        assert U.older_data is not None
        U00 = U.older_data
        U00_f = (
            geo.w_s[:, None] * U00[geo.owner_s]
            + (1.0 - geo.w_s)[:, None] * U00[geo.neighbour_s]
        )
        older_correction = phi.older_single_data - torch.sum(
            U00_f * geo.Sf_s, dim=1
        )
        history_correction = history_correction - c * older_correction

    eps = torch.finfo(phi_corr.dtype).tiny
    coupling = 1.0 - torch.clamp(phi_corr.abs() / (phi0.abs() + eps), max=1.0)
    return coupling * history_correction / grid.dt

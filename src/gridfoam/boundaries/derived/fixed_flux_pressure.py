from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from jaxtyping import Bool, Float, Int

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.utils import get_mask
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.name import make_field_name
from gridfoam.core.shapes import broadcast_entity
from gridfoam.meta.enums import (
    BoundaryConditionType,
    DomainBoundaryPatch,
    FaceSide,
)
from gridfoam.meta.types import PatchName

if TYPE_CHECKING:
    from gridfoam.core.field import CellField, FaceField, GeometricField
else:
    CellField = Any
    FaceField = Any
    GeometricField = Any


def _face_block(
    face_field: FaceField, patch_name: PatchName, side: FaceSide
) -> Float[torch.Tensor, " F_any *component_shape"]:
    """Storage block of ``face_field`` for ``patch_name`` / ``side``."""
    if isinstance(patch_name, DomainBoundaryPatch):
        return face_field.domain_bnd_data
    if side == FaceSide.UPPER:
        return face_field.immersed_upper
    if side == FaceSide.LOWER:
        return face_field.immersed_lower
    raise ValueError(f"Invalid side: {side}")


def _outward_patch_geometry(
    grid: IGridBase,
    patch_name: PatchName,
    side: FaceSide,
    mask: Bool[torch.Tensor, " F_any"],
) -> (
    tuple[
        Int[torch.Tensor, " F_patch"],
        Float[torch.Tensor, " F_patch 3"],
        Float[torch.Tensor, " F_patch"],
    ]
    | None
):
    """
    Adjacent cells, outward ``Sf`` and ``|d|`` for a boundary patch block.

    Returns ``None`` when the grid cannot provide immersed geometry.
    """
    if isinstance(patch_name, DomainBoundaryPatch):
        target_cells = grid.domain_bnd_owner[mask]
        Sf_out = grid.domain_bnd_Sf[mask]
        mag_d = torch.linalg.vector_norm(
            grid.domain_bnd_face_centers[mask]
            - grid.cell_centers[target_cells],
            dim=1,
        )
        return target_cells, Sf_out, mag_d

    if not isinstance(grid, AxisProjectedGrid):
        return None

    immersed_mask = grid.ap_is_immersed_faces
    if side == FaceSide.UPPER:
        target_cells = grid.owner[immersed_mask][mask]
        Sf_out = grid.Sf[immersed_mask][mask]
        mag_d = grid.ap_dist_owner_to_bnd[mask]
    elif side == FaceSide.LOWER:
        target_cells = grid.neighbour[immersed_mask][mask]
        Sf_out = -grid.Sf[immersed_mask][mask]
        mag_d = grid.ap_dist_neighbour_to_bnd[mask]
    else:
        raise ValueError(f"Invalid side: {side}")
    return target_cells, Sf_out, mag_d


def _boundary_face_value(
    field: CellField,
    patch_name: PatchName,
    side: FaceSide,
    target_cells: Int[torch.Tensor, " F_patch"],
    mag_d: Float[torch.Tensor, " F_patch"],
) -> Float[torch.Tensor, " F_patch *component_shape"] | None:
    """
    Face value implied by ``field``'s boundary condition on this patch.

    Uses the same blend as OpenFOAM ``fvPatchField`` /
    ``evaluate_boundary_state``:
    ``psi_b = f * ref_v + (1 - f) * (psi_O + ref_g * |d|)``.
    """
    bc = field.bcs.get(patch_name)
    if bc is None:
        return None
    fraction, ref_v, ref_g = bc.evaluate(field, patch_name, side=side)
    psi_O = field.data[target_cells]
    f = broadcast_entity(fraction, psi_O)
    distance = broadcast_entity(mag_d, psi_O)
    neumann_value = psi_O + distance * ref_g
    return f * ref_v + (1.0 - f) * neumann_value


def _safe_sn_grad(
    flux_pred: Float[torch.Tensor, " F_patch"],
    flux_target: Float[torch.Tensor, " F_patch"],
    coeff: Float[torch.Tensor, " F_patch"],
    mag_Sf: Float[torch.Tensor, " F_patch"],
) -> Float[torch.Tensor, " F_patch"]:
    """``(flux_pred - flux_target) / (coeff * |Sf|)`` with zero-safe denom."""
    denom = coeff * mag_Sf
    return torch.where(
        denom > 0.0,
        (flux_pred - flux_target) / torch.where(denom > 0.0, denom, 1.0),
        torch.zeros_like(denom),
    )


class FixedFluxPressure(BoundaryCondition):
    """
    Fixed-flux pressure boundary condition.

    Sets the pressure normal gradient so that the corrected boundary flux
    equals the velocity boundary flux, as OpenFOAM ``constrainPressure``:

    ``snGrad(p) = (phiHbyA_b - U_b & Sf) / (|Sf| rAtU_b)``.

    ``phiHbyA`` is the predicted flux registered by the pressure-velocity
    algorithm (including ``constrainHbyA``, ``adjustPhi``, ``ddtCorr`` and
    SIMPLEC contributions) and ``rAtU`` the coefficient used in the pressure
    Laplacian (``rAU`` when SIMPLEC is disabled). When ``phiHbyA`` has not
    been registered yet, the predicted flux falls back to the extrapolated
    ``HbyA`` cell value; when no velocity field is available, the current
    boundary ``phi`` is used as the target flux.

    Parameters
    ----------
    phase : str | None, optional
        Phase suffix used to resolve the registered field names.
    """

    def __init__(
        self,
        phase: str | None = None,
    ):
        self.phi_name = make_field_name("phi", phase=phase)
        self.phi_hbya_name = make_field_name("phiHbyA", phase=phase)
        self.HbyA_name = make_field_name("HbyA", phase=phase)
        self.rAU_name = make_field_name("rAU", phase=phase)
        self.rAtU_name = make_field_name("rAtU", phase=phase)
        self.U_name = make_field_name("U", phase=phase)

    @property
    def type(self) -> BoundaryConditionType:
        return BoundaryConditionType.FIXED_FLUX_PRESSURE

    def _coefficient_field(self, grid: IGridBase) -> CellField | None:
        rAtU = grid.get_cellfield(self.rAtU_name)
        if rAtU is not None:
            return rAtU
        return grid.get_cellfield(self.rAU_name)

    def dependencies(self, field: CellField) -> tuple[GeometricField, ...]:
        grid = field.grid
        deps: list[GeometricField] = []
        for candidate in (
            grid.get_facefield(self.phi_hbya_name),
            grid.get_facefield(self.phi_name),
            grid.get_cellfield(self.HbyA_name),
            self._coefficient_field(grid),
            grid.get_cellfield(self.U_name),
        ):
            if candidate is not None:
                deps.append(candidate)
        return tuple(deps)

    def evaluate(
        self,
        field: CellField,
        patch_name: PatchName,
        side: FaceSide = FaceSide.UPPER,
    ) -> tuple[
        Float[torch.Tensor, " F_patch"],
        Float[torch.Tensor, " F_patch *component_shape"],
        Float[torch.Tensor, " F_patch *component_shape"],
    ]:
        if field.component_shape != ():
            raise ValueError("FixedFluxPressure requires a scalar field")
        grid = field.grid
        mask = get_mask(grid, patch_name, side)
        n_faces = int(mask.sum().item())

        # Neumann-style BC: fraction is always zero.
        fraction = torch.zeros((n_faces,), dtype=grid.dtype, device=grid.device)
        ref_v = torch.zeros(
            (n_faces, *field.component_shape),
            dtype=grid.dtype,
            device=grid.device,
        )
        ref_g = torch.zeros_like(ref_v)

        phi = grid.get_facefield(self.phi_name)
        phi_hbya = grid.get_facefield(self.phi_hbya_name)
        HbyA = grid.get_cellfield(self.HbyA_name)
        coeff = self._coefficient_field(grid)
        U = grid.get_cellfield(self.U_name)

        # Recover gradient only when the algorithm has registered the
        # coupling fields. During initialization this falls back to
        # zero gradient.
        if coeff is None or (phi_hbya is None and HbyA is None):
            return fraction, ref_v, ref_g

        geometry = _outward_patch_geometry(grid, patch_name, side, mask)
        if geometry is None:
            return fraction, ref_v, ref_g
        target_cells, Sf_out, mag_d = geometry
        mag_Sf = torch.linalg.vector_norm(Sf_out, dim=1)

        if phi_hbya is not None:
            flux_pred = _face_block(phi_hbya, patch_name, side)[mask]
        else:
            assert HbyA is not None
            flux_pred = torch.sum(HbyA.data[target_cells] * Sf_out, dim=1)

        flux_target = None
        if U is not None:
            psi_b = _boundary_face_value(
                U, patch_name, side, target_cells, mag_d
            )
            if psi_b is not None:
                flux_target = torch.sum(psi_b * Sf_out, dim=1)
        if flux_target is None:
            if phi is None:
                return fraction, ref_v, ref_g
            flux_target = _face_block(phi, patch_name, side)[mask]

        # Before the first pressure corrector ``rAtU`` is still zero; keep
        # the zero-gradient fallback there instead of dividing by zero.
        grad_n = _safe_sn_grad(
            flux_pred, flux_target, coeff.data[target_cells], mag_Sf
        )
        ref_g = grad_n
        return fraction, ref_v, ref_g

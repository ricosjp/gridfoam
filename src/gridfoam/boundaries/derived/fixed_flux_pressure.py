from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.utils import get_mask
from gridfoam.core.builtins import make_builtin_key
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.enums import (
    BoundaryConditionType,
    DomainBoundaryPatch,
    FaceSide,
)
from gridfoam.meta.types import PatchName


class FixedFluxPressure(BoundaryCondition):
    """
    Fixed-flux pressure boundary condition.

    Enforces mass-conservative pressure Neumann values by recovering
    the required pressure gradient from the prescribed face flux.

    Parameters
    ----------
    phi_builtin_key : str, optional
        Builtin key to lookup the face flux field.
    HbyA_builtin_key : str, optional
        Builtin key to lookup the HbyA field.
    rAU_builtin_key : str, optional
        Builtin key to lookup the rAU field.
    """

    def __init__(
        self,
        phi_builtin_key: str | None = None,
        HbyA_builtin_key: str | None = None,
        rAU_builtin_key: str | None = None,
    ):
        self.phi_builtin_key = (
            phi_builtin_key or make_builtin_key("phi", scope="global")
        )
        self.HbyA_builtin_key = (
            HbyA_builtin_key or make_builtin_key("HbyA", scope="global")
        )
        self.rAU_builtin_key = (
            rAU_builtin_key or make_builtin_key("rAU", scope="global")
        )

    @property
    def type(self) -> BoundaryConditionType:
        return BoundaryConditionType.FIXED_FLUX_PRESSURE

    def component(self, c: int) -> BoundaryCondition:
        return self

    def evaluate(
        self,
        field: CellField,
        patch_name: PatchName,
        side: FaceSide = FaceSide.UPPER,
    ) -> tuple[
        Float[torch.Tensor, " F_patch 1"],
        Float[torch.Tensor, " F_patch k"],
        Float[torch.Tensor, " F_patch k"],
    ]:
        grid = field.grid

        # Lookup required fields from the builtin-field registry.
        phi = grid.get_builtin_field(self.phi_builtin_key)
        HbyA = grid.get_builtin_field(self.HbyA_builtin_key)
        rAU = grid.get_builtin_field(self.rAU_builtin_key)

        mask = get_mask(grid, patch_name, side)
        n_faces = mask.sum().item()

        # This boundary behaves as Neumann, so fraction is always zero.
        fraction = torch.zeros(
            (n_faces, 1), dtype=grid.dtype, device=grid.device
        )
        ref_v = torch.zeros(
            (n_faces, field.num_components),
            dtype=grid.dtype,
            device=grid.device,
        )
        ref_g = torch.zeros_like(ref_v)

        # Recover gradient only when all required fields are available.
        # During initialization this naturally falls back to zero gradient.
        if (
            isinstance(phi, FaceField)
            and isinstance(HbyA, CellField)
            and isinstance(rAU, CellField)
        ):
            if isinstance(patch_name, DomainBoundaryPatch):
                phi_bnd = phi.domain_bnd_data[mask]
                target_cells = grid.domain_bnd_owner[mask]
                Sf_bnd = grid.domain_bnd_Sf[mask]
                mag_Sf = torch.linalg.vector_norm(Sf_bnd, dim=1, keepdim=True)
            elif isinstance(grid, AxisProjectedGrid):
                immersed_mask = grid.ap_is_immersed_faces
                if side == FaceSide.UPPER:
                    phi_bnd = phi.immersed_upper[mask]
                    target_cells = grid.owner[immersed_mask][mask]
                elif side == FaceSide.LOWER:
                    phi_bnd = phi.immersed_lower[mask]
                    target_cells = grid.neighbour[immersed_mask][mask]
                else:
                    raise ValueError(f"Invalid side: {side}")
                Sf_bnd = grid.Sf[immersed_mask][mask]
                mag_Sf = torch.linalg.vector_norm(Sf_bnd, dim=1, keepdim=True)

            HbyA_O = HbyA.data[target_cells]
            rAU_O = rAU.data[target_cells]

            # Boundary HbyA flux reconstructed from owner-cell values.
            HbyA_bnd = torch.sum(HbyA_O * Sf_bnd, dim=1, keepdim=True)

            # grad_n = (HbyA_bnd - phi_bnd) / (rAU * |Sf|)
            grad_n = (HbyA_bnd - phi_bnd) / (rAU_O * mag_Sf)
            ref_g = grad_n.expand(-1, field.num_components)

        return fraction, ref_v, ref_g

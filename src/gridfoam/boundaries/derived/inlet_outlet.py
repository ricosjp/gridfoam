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


class InletOutletBC(BoundaryCondition):
    """
    Inlet-outlet boundary condition.

    Behaves as zero-gradient (Neumann) for outflow (phi > 0), and
    switches to fixed-value (Dirichlet) for reverse flow (phi < 0).

    Parameters
    ----------
    inlet_value : Float[torch.Tensor, " k"]
        Fixed value applied during reverse inflow.
    phi_builtin_key : str, optional
        Builtin key to lookup the face flux field.
    """

    def __init__(
        self,
        inlet_value: Float[torch.Tensor, " k"],
        phi_builtin_key: str | None = None,
    ):
        self.inlet_value = inlet_value
        self.phi_builtin_key = phi_builtin_key or make_builtin_key(
            "phi", scope="global"
        )

    @property
    def type(self) -> BoundaryConditionType:
        return BoundaryConditionType.INLET_OUTLET

    def component(self, c: int) -> BoundaryCondition:
        return InletOutletBC(
            self.inlet_value[c],
            self.phi_builtin_key,
        )

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
        phi = grid.get_builtin_field(self.phi_builtin_key)
        if not isinstance(phi, FaceField):
            raise ValueError(
                f"Builtin field {self.phi_builtin_key} is not a FaceField."
            )

        mask = get_mask(grid, patch_name, side)
        n_faces = int(mask.sum().item())

        # Default behavior is Neumann (outflow).
        fraction = torch.zeros(
            (n_faces, 1), dtype=grid.dtype, device=grid.device
        )
        ref_v = torch.zeros(
            (n_faces, field.num_components),
            dtype=grid.dtype,
            device=grid.device,
        )
        ref_g = torch.zeros_like(ref_v)

        # Return zero-gradient behavior when there are no faces.
        if n_faces == 0:
            return fraction, ref_v, ref_g

        # Fetch local face flux values on the target boundary.
        if isinstance(patch_name, DomainBoundaryPatch):
            phi_bnd = phi.domain_bnd_data[mask]

        elif isinstance(grid, AxisProjectedGrid):
            if side == FaceSide.UPPER:
                phi_bnd = phi.immersed_upper[mask]
            elif side == FaceSide.LOWER:
                phi_bnd = phi.immersed_lower[mask]
            else:
                raise ValueError(f"Invalid side: {side}")
        else:
            raise NotImplementedError(
                "IBM for this grid type is not supported yet."
            )

        # Detect inflow region (phi < 0).
        is_inflow = phi_bnd[:, 0] < 0.0

        # fraction: 1.0 for inflow (Dirichlet), 0.0 for outflow (Neumann).
        fraction[is_inflow] = 1.0

        # Apply configured inlet value on inflow faces.
        inlet_value = self.inlet_value.to(
            dtype=grid.dtype,
            device=grid.device,
        )
        ref_v[is_inflow] = inlet_value[None, :]

        return fraction, ref_v, ref_g

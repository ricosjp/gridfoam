from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from jaxtyping import Float

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.utils import get_mask_and_size
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.name import make_field_name
from gridfoam.core.shapes import require_shape
from gridfoam.meta.enums import (
    BoundaryConditionType,
    DomainBoundaryPatch,
    FaceSide,
)
from gridfoam.meta.types import PatchName

if TYPE_CHECKING:
    from gridfoam.core.field import CellField, GeometricField
else:
    CellField = Any
    GeometricField = Any


class InletOutletBC(BoundaryCondition):
    """
    Inlet-outlet boundary condition.

    Uses zero gradient for outward or zero flux (``phi >= 0``), and the
    prescribed value for reverse inflow (``phi < 0``). This condition owns
    the switching; convection operators use its evaluated value fraction.

    Parameters
    ----------
    inlet_value : torch.Tensor
        Uniform reverse-inflow value with shape ``component_shape``;
        a scalar uses shape ``()``.
    phi_builtin_key : str, optional
        Builtin key to lookup the face flux field.
    """

    def __init__(
        self,
        inlet_value: Float[torch.Tensor, " *component_shape"],
        phase: str | None = None,
    ):
        self.inlet_value = inlet_value
        self.phase = phase
        self.phi_name = make_field_name("phi", phase=phase)

    @property
    def type(self) -> BoundaryConditionType:
        return BoundaryConditionType.INLET_OUTLET

    def dependencies(self, field: CellField) -> tuple[GeometricField, ...]:
        phi = field.grid.get_facefield(self.phi_name)
        return () if phi is None else (phi,)

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
        require_shape(self.inlet_value, field.component_shape, "inlet value")
        grid = field.grid

        mask, n_faces = get_mask_and_size(grid, patch_name, side)

        # Default behavior is Neumann (outflow).
        fraction = torch.zeros((n_faces,), dtype=grid.dtype, device=grid.device)
        ref_v = torch.zeros(
            (n_faces, *field.component_shape),
            dtype=grid.dtype,
            device=grid.device,
        )
        ref_g = torch.zeros_like(ref_v)

        # Return zero-gradient behavior when there are no faces.
        if n_faces == 0:
            return fraction, ref_v, ref_g

        # Lookup required fields from the field registry.
        phi = grid.get_facefield(self.phi_name)
        if phi is None:
            raise ValueError(f"Field {self.phi_name} is not found.")

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
        is_inflow = phi_bnd < 0.0

        # fraction: 1.0 for inflow (Dirichlet), 0.0 for outflow (Neumann).
        fraction[is_inflow] = 1.0

        # Apply configured inlet value on inflow faces.
        inlet_value = self.inlet_value.to(
            dtype=grid.dtype,
            device=grid.device,
        )
        ref_v[is_inflow] = inlet_value

        return fraction, ref_v, ref_g

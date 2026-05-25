from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.fv.boundary_ops import (
    BoundaryBatch,
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)
from gridfoam.meta.types import PatchName
from gridfoam.models.turbulence.base import TurbulenceModel


@dataclass(frozen=True)
class ForceCoefficients:
    force: Float[torch.Tensor, " 3"]
    pressure_force: Float[torch.Tensor, " 3"]
    viscous_force: Float[torch.Tensor, " 3"]
    coefficient: Float[torch.Tensor, " 3"]
    pressure_coefficient: Float[torch.Tensor, " 3"]
    viscous_coefficient: Float[torch.Tensor, " 3"]
    cd: Float[torch.Tensor, ""]
    pressure_cd: Float[torch.Tensor, ""]
    viscous_cd: Float[torch.Tensor, ""]


def compute_force_coefficients(
    grid: IGridBase,
    U: CellField,
    p: CellField,
    turbulence: TurbulenceModel,
    *,
    patches: Iterable[PatchName] | None = None,
    drag_direction: Sequence[float] = (1.0, 0.0, 0.0),
    reference_velocity: float,
    reference_area: float,
) -> ForceCoefficients:
    """
    Compute force coefficients on immersed-boundary patches.

    The pressure term uses the pressure boundary value. The viscous term uses
    the tangential part of ``snGrad(U)`` at the immersed boundary:
    ``F_v = -nu_eff * A * snGrad(U)_t``. This matches the incompressible
    pressure convention used by the solver, where ``p`` is kinematic pressure.
    """
    _validate_inputs(grid, U, p, reference_velocity, reference_area)
    assert isinstance(grid, AxisProjectedGrid)

    patch_filter = set(patches) if patches is not None else None
    drag_dir = _unit_vector(
        drag_direction, dtype=grid.dtype, device=grid.device
    )
    coeff_denom = 0.5 * reference_velocity**2 * reference_area

    pressure_force = torch.zeros(3, dtype=grid.dtype, device=grid.device)
    viscous_force = torch.zeros(3, dtype=grid.dtype, device=grid.device)
    nu_eff = turbulence.nu_eff()

    for p_batch in iter_boundary_batches(p):
        if not _is_target_immersed_patch(p_batch, patch_filter):
            continue

        U_batch = _matching_velocity_batch(U, p_batch)
        p_b = evaluate_boundary_state(p, p_batch)[3]
        dUdn = _boundary_sn_grad(U, U_batch)

        target_cells = p_batch.target_cells
        face_area_vector = _immersed_face_area_vector(grid, p_batch)
        area = torch.linalg.vector_norm(face_area_vector, dim=1, keepdim=True)
        normal = face_area_vector / area

        dUdn_normal = torch.sum(dUdn * normal, dim=1, keepdim=True) * normal
        dUdn_tangential = dUdn - dUdn_normal

        pressure_force = pressure_force + torch.sum(
            p_b * face_area_vector, dim=0
        )
        viscous_force = viscous_force - torch.sum(
            nu_eff[target_cells] * area * dUdn_tangential, dim=0
        )

    force = pressure_force + viscous_force
    coefficient = force / coeff_denom
    pressure_coefficient = pressure_force / coeff_denom
    viscous_coefficient = viscous_force / coeff_denom

    return ForceCoefficients(
        force=force,
        pressure_force=pressure_force,
        viscous_force=viscous_force,
        coefficient=coefficient,
        pressure_coefficient=pressure_coefficient,
        viscous_coefficient=viscous_coefficient,
        cd=torch.dot(coefficient, drag_dir),
        pressure_cd=torch.dot(pressure_coefficient, drag_dir),
        viscous_cd=torch.dot(viscous_coefficient, drag_dir),
    )


def _validate_inputs(
    grid: IGridBase,
    U: CellField,
    p: CellField,
    reference_velocity: float,
    reference_area: float,
) -> None:
    if not isinstance(grid, AxisProjectedGrid):
        raise TypeError("force coefficients require an AxisProjectedGrid.")
    if U.grid is not grid or p.grid is not grid:
        raise ValueError("U and p must belong to the supplied grid.")
    if U.num_components != 3:
        raise ValueError("U must be a 3-component vector field.")
    if p.num_components != 1:
        raise ValueError("p must be a scalar field.")
    if reference_velocity <= 0.0:
        raise ValueError("reference_velocity must be positive.")
    if reference_area <= 0.0:
        raise ValueError("reference_area must be positive.")


def _unit_vector(
    values: Sequence[float], *, dtype: torch.dtype, device: torch.device
) -> Float[torch.Tensor, " 3"]:
    direction = torch.tensor(values, dtype=dtype, device=device)
    if direction.shape != (3,):
        raise ValueError("drag_direction must contain exactly 3 values.")
    norm = torch.linalg.vector_norm(direction)
    if norm <= 0.0:
        raise ValueError("drag_direction must be non-zero.")
    return direction / norm


def _is_target_immersed_patch(
    batch: BoundaryBatch, patch_filter: set[PatchName] | None
) -> bool:
    if batch.face_kind not in (
        BoundaryFaceKind.IMMERSED_UPPER,
        BoundaryFaceKind.IMMERSED_LOWER,
    ):
        return False
    return patch_filter is None or batch.patch_name in patch_filter


def _matching_velocity_batch(
    U: CellField, p_batch: BoundaryBatch
) -> BoundaryBatch:
    for U_batch in iter_boundary_batches(U):
        if (
            U_batch.patch_name == p_batch.patch_name
            and U_batch.face_kind == p_batch.face_kind
        ):
            return U_batch
    raise ValueError(
        f"U is missing boundary condition for patch {p_batch.patch_name!r}."
    )


def _boundary_sn_grad(
    field: CellField, batch: BoundaryBatch
) -> Float[torch.Tensor, " F_any k"]:
    psi_b = evaluate_boundary_state(field, batch)[3]
    psi_O = field.data[batch.target_cells]
    return (psi_b - psi_O) / batch.mag_d


def _immersed_face_area_vector(
    grid: AxisProjectedGrid, batch: BoundaryBatch
) -> Float[torch.Tensor, " F_any 3"]:
    immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
    if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
        return immersed_Sf[batch.face_mask]
    if batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
        return -immersed_Sf[batch.face_mask]
    raise ValueError("batch must be an immersed-boundary batch.")

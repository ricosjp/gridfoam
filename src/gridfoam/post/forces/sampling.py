from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.enums import FaceSide


@dataclass(frozen=True)
class SurfaceSample:
    """AP-side sampled boundary data anchored to surface-mesh faces."""

    anchor_id: Int[torch.Tensor, " F_sample"]
    n_hat: Float[torch.Tensor, " F_sample 3"]
    Sf: Float[torch.Tensor, " F_sample 3"]
    p_b: Float[torch.Tensor, " F_sample 1"]
    U_b: Float[torch.Tensor, " F_sample 3"]
    U_cell: Float[torch.Tensor, " F_sample 3"]
    nu_eff: Float[torch.Tensor, " F_sample 1"]
    mag_d: Float[torch.Tensor, " F_sample 1"]
    face_centers: Float[torch.Tensor, " F_sample 3"]


def boundary_value(
    field: CellField,
    patch_name: str,
    side: FaceSide,
    target_cells: Int[torch.Tensor, " F_patch"],
    mag_d: Float[torch.Tensor, " F_patch 1"],
) -> Float[torch.Tensor, " F_patch k"]:
    """
    Evaluate a boundary value using the field's value-fraction form.

    Parameters
    ----------
    field : CellField
        Field whose boundary state is requested.
    patch_name : str
        Patch name.
    side : FaceSide
        AP-IBM face side.
    target_cells : torch.Tensor
        Adjacent fluid-cell ids.
    mag_d : torch.Tensor
        Distance from cell center to boundary along the AP axis.

    Returns
    -------
    torch.Tensor
        Boundary values on the requested patch side.
    """
    if patch_name not in field.bcs:
        return field.data[target_cells]

    fraction, ref_v, ref_g = field.bcs[patch_name].evaluate(
        field, patch_name, side=side
    )
    psi_O = field.data[target_cells]
    return fraction * ref_v + (1.0 - fraction) * (psi_O + ref_g * mag_d)


def sample_side(
    *,
    p: CellField,
    U: CellField,
    nu_eff: Float[torch.Tensor, " C 1"],
    patch_name: str,
    side: FaceSide,
    target_cells: Int[torch.Tensor, " F_patch"],
    Sf: Float[torch.Tensor, " F_patch 3"],
    mag_d: Float[torch.Tensor, " F_patch 1"],
    anchor_id: Int[torch.Tensor, " F_patch"],
    cell_centers: Float[torch.Tensor, " F_patch 3"],
) -> SurfaceSample:
    """
    Evaluate AP-side boundary samples for later surface-face integration.

    Parameters
    ----------
    p : CellField
        Kinematic pressure field.
    U : CellField
        Velocity field.
    nu_eff : torch.Tensor
        Effective kinematic viscosity per cell.
    patch_name : str
        Immersed patch name.
    side : FaceSide
        Owner-side or neighbour-side boundary.
    target_cells : torch.Tensor
        Fluid cells adjacent to the immersed boundary faces.
    Sf : torch.Tensor
        AP-projected face area vectors.
    mag_d : torch.Tensor
        AP-axis distance from cell center to boundary.
    anchor_id : torch.Tensor
        Surface-mesh face ids associated with these AP samples.
    cell_centers : torch.Tensor
        Adjacent-cell centers for AP boundary-point reconstruction.
    """
    p_b = boundary_value(p, patch_name, side, target_cells, mag_d)
    U_b = boundary_value(U, patch_name, side, target_cells, mag_d)

    mag_Sf = torch.linalg.vector_norm(Sf, dim=1, keepdim=True)
    n_hat = Sf / mag_Sf

    return SurfaceSample(
        anchor_id=anchor_id,
        n_hat=n_hat,
        Sf=Sf,
        p_b=p_b,
        U_b=U_b,
        U_cell=U.data[target_cells],
        nu_eff=nu_eff[target_cells],
        mag_d=mag_d,
        face_centers=cell_centers + n_hat * mag_d,
    )


def iter_patch_side_samples(
    grid: AxisProjectedGrid,
    *,
    p: CellField,
    U: CellField,
    nu_eff: Float[torch.Tensor, " C 1"],
    patch_name: str,
) -> Iterable[SurfaceSample]:
    immersed_owner = grid.owner[grid.ap_is_immersed_faces]
    immersed_neighbour = grid.neighbour[grid.ap_is_immersed_faces]
    immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
    upper_mask, lower_mask = grid.ap_get_patch_mask(patch_name)

    if torch.any(upper_mask):
        yield sample_side(
            p=p,
            U=U,
            nu_eff=nu_eff,
            patch_name=patch_name,
            side=FaceSide.UPPER,
            target_cells=immersed_owner[upper_mask],
            Sf=immersed_Sf[upper_mask],
            mag_d=grid.ap_dist_owner_to_bnd[upper_mask],
            anchor_id=grid.ap_owner_bnd_anchor_id[upper_mask],
            cell_centers=grid.cell_centers[immersed_owner[upper_mask]],
        )

    if torch.any(lower_mask):
        yield sample_side(
            p=p,
            U=U,
            nu_eff=nu_eff,
            patch_name=patch_name,
            side=FaceSide.LOWER,
            target_cells=immersed_neighbour[lower_mask],
            Sf=-immersed_Sf[lower_mask],
            mag_d=grid.ap_dist_neighbour_to_bnd[lower_mask],
            anchor_id=grid.ap_neighbour_bnd_anchor_id[lower_mask],
            cell_centers=grid.cell_centers[immersed_neighbour[lower_mask]],
        )

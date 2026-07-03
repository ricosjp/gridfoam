"""Unit tests for ``FvMatrix.flux``."""

from __future__ import annotations

import torch

from gridfoam.algorithms.utils.pressure_correction import correct_phi_inconsistent
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv import fvc, fvm
from gridfoam.fv.fvc.interpolate import single_internal_mask
from gridfoam.meta.enums import FieldRole


def test_fvmatrix_flux_matches_sn_grad_laplacian(
    small_axis_projected_grid: AxisProjectedGrid,
):
    # Laplacian matrix flux must equal gamma_f * |Sf| * snGrad(psi) on
    # single internal faces (gamma = 1).
    grid = small_axis_projected_grid
    psi = CellField(grid, "psi_flux", role=FieldRole.LOCAL, num_components=1)
    torch.manual_seed(0)
    psi.data = torch.randn_like(psi.data)

    mat = fvm.laplacian(1.0, psi)
    flux = mat.flux(psi.data)

    sn_grad = fvc.sn_grad(psi).single_data
    single_mask = single_internal_mask(grid)
    mag_Sf = torch.linalg.vector_norm(grid.Sf[single_mask], dim=1, keepdim=True)
    expected = mag_Sf * sn_grad

    assert flux.shape == expected.shape
    assert torch.allclose(flux, expected, rtol=1e-5, atol=1e-6)


def test_negative_laplacian_flux_matches_pressure_equation_correction(
    small_axis_projected_grid: AxisProjectedGrid,
):
    # (-pEqn).flux must match the snGrad-based inconsistent phi correction
    # used in pressure-correction algorithms.
    grid = small_axis_projected_grid
    p = CellField(grid, "p_neg_flux", role=FieldRole.LOCAL, num_components=1)
    rAU = CellField(grid, "rAU_neg_flux", role=FieldRole.LOCAL, num_components=1)
    phi = FaceField(grid, "phi_neg_flux", role=FieldRole.LOCAL, num_components=1)

    torch.manual_seed(1)
    p.data = torch.randn_like(p.data)
    rAU.data = torch.rand_like(rAU.data) + 0.1
    phi_hbya = torch.randn(
        (int(phi.single_mask.sum().item()), 1),
        dtype=grid.dtype,
        device=grid.device,
    )

    p_eqn_mat = -fvm.laplacian(rAU.data, p)
    flux_from_matrix = (-p_eqn_mat).flux(p.data)

    correct_phi_inconsistent(phi, rAU, p, phi_hbya)
    phi_from_inconsistent = phi.single_data.clone()
    phi_from_matrix = phi_hbya - flux_from_matrix

    assert torch.allclose(
        phi_from_matrix, phi_from_inconsistent, rtol=1e-4, atol=1e-6
    )


def test_fvmatrix_flux_zero_on_immersed_coefficients(
    small_axis_projected_grid: AxisProjectedGrid,
):
    # Immersed-boundary faces must have zero upper coefficients and finite
    # flux output on the remaining single internal faces.
    grid = small_axis_projected_grid
    if not torch.any(grid.ap_is_immersed_faces):
        return

    p = CellField(grid, "p_imm", role=FieldRole.LOCAL, num_components=1)
    mat = fvm.laplacian(1.0, p)
    assert torch.all(mat.upper[grid.ap_is_immersed_faces] == 0.0)

    flux = mat.flux(p.data)
    single_mask = single_internal_mask(grid)
    assert flux.shape[0] == int(single_mask.sum().item())
    assert torch.all(torch.isfinite(flux))

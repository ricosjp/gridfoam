"""
Integration tests for ``fvm.laplacian`` scheme selection on refined meshes.

Pins that ``corrected`` is exact for linear fields on hanging faces, matches
``fvc.sn_grad``, and that ``uncorrected`` / scheme lookup behave as configured.
"""

from __future__ import annotations

import torch
from tests.helpers import linear_scalar_field, refined_3d_grid, refined_grid

from gridfoam.core.field import CellField
from gridfoam.core.grid.base import IGridBase
from gridfoam.fv import fvc, fvm
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.fv.kernels.face_interpolation import linear_internal_face_values
from gridfoam.meta.config import fvSchemesConfig
from gridfoam.meta.enums import FieldRole, GradScheme, LaplacianScheme


def _exact_flux(grid: IGridBase, gradient: torch.Tensor) -> torch.Tensor:
    geo = face_geometry(grid)
    return geo.mag_Sf_s * gradient[grid.axis[geo.single_idx]]


def test_corrected_laplacian_flux_is_exact_for_linear_field():
    # gamma * |Sf| * snGrad must be exact on every single-sided face,
    # including hanging faces, with either configured gradient scheme.
    for grad_scheme in (GradScheme.LINEAR, GradScheme.LEASTSQUARE):
        grid = refined_3d_grid(grad_scheme)
        field, gradient = linear_scalar_field(grid)

        mat = fvm.laplacian(1.0, field)
        flux = mat.flux(field.data)

        torch.testing.assert_close(
            flux, _exact_flux(grid, gradient), atol=1e-12, rtol=1e-12
        )
        assert mat.face_flux_correction is not None


def test_corrected_laplacian_matches_sn_grad_path_on_refined_mesh():
    # Matrix flux must equal gamma_f * |Sf| * snGrad for non-uniform gamma,
    # so correct_phi via (-pEqn).flux stays consistent with snGrad.
    grid = refined_grid()
    geo = face_geometry(grid)
    psi = CellField(grid, "psi_rand", FieldRole.LOCAL, ())
    torch.manual_seed(3)
    psi.data = torch.randn_like(psi.data)
    gamma = CellField(grid, "gamma_rand", FieldRole.LOCAL, ())
    gamma.data = torch.rand_like(psi.data) + 0.5

    flux = fvm.laplacian(gamma.data, psi).flux(psi.data)
    gamma_f = linear_internal_face_values(gamma, geo)
    expected = gamma_f * geo.mag_Sf_s * fvc.sn_grad(psi).single_data

    torch.testing.assert_close(flux, expected, atol=1e-12, rtol=1e-12)


def test_uncorrected_laplacian_has_no_face_flux_correction():
    # ``uncorrected`` keeps the orthogonal two-point stencil only: exact on
    # regular faces, O(1) normal-gradient error on hanging faces.
    grid = refined_grid(
        fv_schemes=fvSchemesConfig(
            laplacianSchemes={"default": LaplacianScheme.UNCORRECTED}
        )
    )
    field, gradient = linear_scalar_field(grid)
    geo = face_geometry(grid)

    mat = fvm.laplacian(1.0, field)
    assert mat.face_flux_correction is None

    flux = mat.flux(field.data)
    expected = _exact_flux(grid, gradient)
    regular = torch.ones(geo.num_single, dtype=torch.bool)
    regular[geo.hang_idx] = False
    torch.testing.assert_close(
        flux[regular], expected[regular], atol=1e-12, rtol=1e-12
    )
    assert (flux[geo.hang_idx] - expected[geo.hang_idx]).abs().max() > 1e-3


def test_linear_laplacian_scheme_is_alias_of_corrected():
    # Legacy ``linear`` value must keep the corrected behaviour.
    grid = refined_grid(
        fv_schemes=fvSchemesConfig(
            laplacianSchemes={"default": LaplacianScheme.LINEAR}
        )
    )
    field, gradient = linear_scalar_field(grid)

    flux = fvm.laplacian(1.0, field).flux(field.data)

    torch.testing.assert_close(
        flux, _exact_flux(grid, gradient), atol=1e-12, rtol=1e-12
    )


def test_laplacian_scheme_lookup_prefers_field_specific_key():
    # ``laplacian(<field>)`` must take precedence over ``default``.
    grid = refined_grid(
        fv_schemes=fvSchemesConfig(
            laplacianSchemes={
                "default": LaplacianScheme.UNCORRECTED,
                "laplacian(psi)": LaplacianScheme.CORRECTED,
            }
        )
    )
    field, gradient = linear_scalar_field(grid)

    flux = fvm.laplacian(1.0, field).flux(field.data)

    torch.testing.assert_close(
        flux, _exact_flux(grid, gradient), atol=1e-12, rtol=1e-12
    )

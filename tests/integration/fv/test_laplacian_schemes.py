"""
Numerical Laplacian schemes on refined meshes.

Guarantees hanging-face flux for ``corrected``, agreement with ``fvc.sn_grad``,
and that ``uncorrected`` / legacy ``linear`` / field-specific YAML actually
change the assembled operator. Lookup *order* is unit-tested in
``test_fv_scheme_selection``.
"""

from __future__ import annotations

import torch
from tests.helpers import linear_scalar_field, refined_3d_grid, refined_grid

from gridfoam.core.field import CellField
from gridfoam.core.grid.base import GridBase
from gridfoam.fv import fvc, fvm
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.fv.kernels.face_interpolation import linear_internal_face_values
from gridfoam.meta.config import fvSchemesConfig
from gridfoam.meta.enums import FieldRole, GradScheme, LaplacianScheme


def _exact_flux(grid: GridBase, gradient: torch.Tensor) -> torch.Tensor:
    geo = face_geometry(grid)
    return geo.mag_Sf_s * gradient[grid.axis[geo.single_idx]]


def test_corrected_laplacian_flux_is_exact_for_linear_field() -> None:
    """Hanging-face flux is exact for a linear field under both grad schemes."""
    for grad_scheme in (GradScheme.LINEAR, GradScheme.LEASTSQUARE):
        grid = refined_3d_grid(grad_scheme)
        field, gradient = linear_scalar_field(grid)

        mat = fvm.laplacian(1.0, field)
        flux = mat.flux(field.data)

        torch.testing.assert_close(
            flux, _exact_flux(grid, gradient), atol=1e-12, rtol=1e-12
        )
        assert mat.face_flux_correction is not None


def test_corrected_laplacian_matches_sn_grad_path_on_refined_mesh() -> None:
    """``fvm.laplacian`` flux must match ``gamma_f * |Sf| * fvc.sn_grad``."""
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


def test_uncorrected_laplacian_has_no_face_flux_correction() -> None:
    """Orthogonal stencil: exact on regular faces, wrong on hanging faces."""
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


def test_linear_yaml_alias_keeps_corrected_hanging_face_flux() -> None:
    """Legacy ``linear`` must remain the corrected stencil on 2:1 faces."""
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


def test_laplacian_operator_uses_the_field_specific_yaml_key() -> None:
    """``fvm.laplacian`` must call scheme lookup, not a hardcoded default."""
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

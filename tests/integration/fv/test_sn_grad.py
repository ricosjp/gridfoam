"""
Integration tests for surface-normal gradient (``sn_grad``) reconstruction.

Scheme selection and hanging-node accuracy on refined octree meshes.
"""

from __future__ import annotations

import torch
from tests.helpers import linear_scalar_field, refined_grid

from gridfoam.fv.fvc.sn_grad import sn_grad
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.meta.config import fvSchemesConfig
from gridfoam.meta.enums import GradScheme, SnGradScheme


def test_sn_grad_corrected_is_exact_for_linear_field():
    # Default ``corrected`` snGrad uses a local least-squares gradient on
    # hanging cells, so it is exact for any configured ``gradSchemes``.
    for grad_scheme in (None, GradScheme.LINEAR, GradScheme.LEASTSQUARE):
        grid = refined_grid(grad_scheme=grad_scheme)
        field, gradient = linear_scalar_field(grid)

        result = sn_grad(field)
        expected = gradient[grid.axis[result.single_mask]].reshape(-1, 1)
        torch.testing.assert_close(
            result.single_data, expected, atol=1e-12, rtol=1e-12
        )


def test_sn_grad_uncorrected_scheme_skips_hanging_correction():
    # ``uncorrected`` is the two-point difference: exact on regular faces,
    # not on hanging-node faces for a skewed field.
    grid = refined_grid(
        fv_schemes=fvSchemesConfig(
            snGradSchemes={"default": SnGradScheme.UNCORRECTED}
        )
    )
    field, gradient = linear_scalar_field(grid)
    geo = face_geometry(grid)

    result = sn_grad(field).single_data
    expected = gradient[grid.axis[geo.single_idx]].reshape(-1, 1)

    regular = torch.ones(geo.num_single, dtype=torch.bool)
    regular[geo.hang_idx] = False
    torch.testing.assert_close(
        result[regular], expected[regular], atol=1e-12, rtol=1e-12
    )
    assert (result[geo.hang_idx] - expected[geo.hang_idx]).abs().max() > 1e-3


def test_sn_grad_scheme_lookup_prefers_field_specific_key():
    # ``snGrad(<field>)`` must take precedence over ``default``.
    grid = refined_grid(
        fv_schemes=fvSchemesConfig(
            snGradSchemes={
                "default": SnGradScheme.UNCORRECTED,
                "snGrad(psi)": SnGradScheme.CORRECTED,
            }
        )
    )
    field, gradient = linear_scalar_field(grid)
    assert field.name == "psi"

    result = sn_grad(field)
    expected = gradient[grid.axis[result.single_mask]].reshape(-1, 1)
    torch.testing.assert_close(
        result.single_data, expected, atol=1e-12, rtol=1e-12
    )

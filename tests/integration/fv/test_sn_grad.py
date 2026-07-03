"""
Integration tests for surface-normal gradient (``sn_grad``) reconstruction.

Exercises scheme selection, least-squares accuracy at octree interfaces, and
reuse of a cached ``grad`` field.
"""

from __future__ import annotations

import importlib

import torch
from pytest import MonkeyPatch

from gridfoam.core.field import CellField
from gridfoam.fv.fvc.grad import grad
from gridfoam.fv.fvc.sn_grad import sn_grad
from gridfoam.meta.enums import GradScheme
from tests.helpers import linear_scalar_field, refined_grid

grad_schemes_module = importlib.import_module("gridfoam.fv.schemes.grad")


def test_sn_grad_default_linear_scheme_does_not_use_leastsquare(
    monkeypatch: MonkeyPatch,
):
    # Default LINEAR ``sn_grad`` must not call least-squares reconstruction
    # (it should use corrected-linear face values instead).
    def fail_leastsquare(_field: CellField) -> torch.Tensor:
        raise AssertionError("leastSquare reconstruction should not be used")

    monkeypatch.setattr(
        grad_schemes_module, "_least_square_grad_data", fail_leastsquare
    )

    grid = refined_grid()
    field, _ = linear_scalar_field(grid)

    sn_grad(field)


def test_sn_grad_is_exact_with_leastsquare_grad_scheme_on_octree_interfaces():
    # With LEASTSQUARE as the default grad scheme, ``sn_grad`` must recover
    # the analytic normal derivative on faces adjacent to octree interfaces.
    grid = refined_grid(grad_scheme=GradScheme.LEASTSQUARE)
    field, gradient = linear_scalar_field(grid)

    sn_grad_result = sn_grad(field)
    expected = gradient[grid.axis[sn_grad_result.single_mask]].reshape(-1, 1)

    torch.testing.assert_close(
        sn_grad_result.single_data, expected, atol=1e-12, rtol=1e-12
    )


def test_sn_grad_reuses_cached_grad_field():
    # A prior ``grad`` call must populate the cache so ``sn_grad`` avoids
    # recomputing cell-centre gradients while staying exact.
    grid = refined_grid(grad_scheme=GradScheme.LEASTSQUARE)
    field_direct, gradient = linear_scalar_field(grid)

    sn_grad_direct = sn_grad(field_direct)

    field_cached, _ = linear_scalar_field(grid)
    grad(field_cached)
    sn_grad_cached = sn_grad(field_cached)

    expected = gradient[grid.axis[sn_grad_direct.single_mask]].reshape(-1, 1)
    torch.testing.assert_close(
        sn_grad_direct.single_data, expected, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        sn_grad_cached.single_data, expected, atol=1e-12, rtol=1e-12
    )

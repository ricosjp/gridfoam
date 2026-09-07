"""Exercise real AMG cycles, warm starts, and component-wise stopping."""

from typing import cast

import numpy as np
import pytest
import torch
from scipy.sparse import csr_array, diags_array

from gridfoam.solvers.pyamg_bridge import (
    _solve_csr_components,  # pyright: ignore[reportPrivateUsage]
)


def _csr_diags(diagonals: list[np.ndarray], offsets: list[int]) -> csr_array:
    # SciPy stubs type ``offsets`` as ``int`` and the CSR ``format`` as DIA.
    return cast(
        csr_array,
        diags_array(
            diagonals,
            offsets=offsets,  # pyright: ignore[reportArgumentType]
            format="csr",
        ),
    )


def system() -> tuple[csr_array, torch.Tensor]:
    n = 64
    matrix = _csr_diags(
        [-np.ones(n - 1), 2.01 * np.ones(n), -np.ones(n - 1)],
        [-1, 0, 1],
    )
    rhs = torch.linspace(1, 2, n, dtype=torch.float64)
    return matrix, rhs


@pytest.mark.parametrize("norm", [2, float("inf")])
@pytest.mark.parametrize("relative", [False, True])
def test_amg_true_residual_and_tolerances(norm: int | float, relative: bool):
    matrix, rhs = system()
    # A nonzero warm start separates ||r0|| from ||b||.
    exact = np.linalg.solve(matrix.toarray(), rhs.numpy())
    x0 = torch.from_numpy(exact) + 0.01
    atol, rtol = (1e-14, 1e-3) if relative else (1e-8, 0.0)
    result = _solve_csr_components(
        matrix,
        rhs,
        x0,
        atol=atol,
        rtol=rtol,
        max_iter=100,
        norm_order=norm,
    )
    initial = np.linalg.norm(rhs.numpy() - matrix @ x0.numpy(), ord=norm)
    final = np.linalg.norm(
        rhs.numpy() - matrix @ result.solution.numpy(), ord=norm
    )
    (stats,) = result.stats
    assert stats.initial_residual == pytest.approx(initial)
    assert stats.final_residual == pytest.approx(final)
    assert stats.converged
    assert 0 < stats.iterations < 100
    assert final < max(atol, rtol * initial)
    assert torch.equal(x0, torch.from_numpy(exact) + 0.01)


def test_amg_reports_iteration_limit_instead_of_success():
    matrix, rhs = system()
    result = _solve_csr_components(
        matrix,
        rhs,
        torch.zeros_like(rhs),
        atol=1e-30,
        rtol=0.0,
        max_iter=1,
        norm_order=2,
    )
    (stats,) = result.stats
    assert stats.iterations == 1
    assert not stats.converged
    assert stats.final_residual > 1e-30


@pytest.mark.parametrize("norm, iterations", [(2, 1), (float("inf"), 0)])
def test_amg_honors_norm_and_skips_converged_components(
    norm: int | float, iterations: int
):
    matrix = _csr_diags([np.ones(16)], [0])
    rhs = torch.zeros(16, 3, dtype=torch.float64)
    x0 = rhs.clone()
    x0[:, 1] = 0.5
    result = _solve_csr_components(
        matrix,
        rhs,
        x0,
        atol=0.75,
        rtol=0.0,
        max_iter=10,
        norm_order=norm,
    )
    assert len(result.stats) == 3
    assert result.stats[0].iterations == 0
    assert result.stats[0].initial_residual == 0
    assert result.stats[1].iterations == iterations
    assert all(s.converged for s in result.stats)


def test_amg_zero_rhs_nonzero_initial_guess():
    matrix, rhs = system()
    rhs.zero_()
    result = _solve_csr_components(
        matrix,
        rhs,
        torch.ones_like(rhs),
        atol=1e-14,
        rtol=1e-5,
        max_iter=100,
        norm_order=2,
    )
    (stats,) = result.stats
    assert stats.converged
    assert stats.iterations > 0
    assert stats.final_residual < 1e-5 * stats.initial_residual


def test_amg_nonfinite_residual_is_not_converged():
    matrix, rhs = system()
    rhs[0] = float("nan")
    result = _solve_csr_components(
        matrix,
        rhs,
        torch.zeros_like(rhs),
        atol=1e-8,
        rtol=0.0,
        max_iter=10,
        norm_order=2,
    )
    assert not result.stats[0].converged
    assert result.stats[0].iterations == 0

from __future__ import annotations

import numpy as np
import pyamg
import torch
from jaxtyping import Float, Int
from scipy.sparse import csr_array

from gridfoam.core.equation import Equation
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.shapes import (
    component_indices,
    require_shape,
    validate_component_shape,
)
from gridfoam.meta.config import SolverConfig
from gridfoam.solvers.base import LinearSolver, SolveResult, SolveStats


def _build_csr(
    diag: Float[torch.Tensor, " C"],
    upper: Float[torch.Tensor, " F"],
    lower: Float[torch.Tensor, " F"],
    owner: Int[torch.Tensor, " F"],
    neighbour: Int[torch.Tensor, " F"],
) -> csr_array:
    """Convert LDU coefficients to a SciPy CSR matrix."""
    n_cells = int(diag.shape[0])
    rows = torch.cat(
        [
            torch.arange(n_cells, device=diag.device, dtype=torch.long),
            owner,
            neighbour,
        ],
        dim=0,
    )
    cols = torch.cat(
        [
            torch.arange(n_cells, device=diag.device, dtype=torch.long),
            neighbour,
            owner,
        ],
        dim=0,
    )
    vals = torch.cat([diag, upper, lower], dim=0)

    A_csr = csr_array(
        (
            vals.detach().cpu().numpy(),
            (
                rows.detach().cpu().numpy().astype(np.int32),
                cols.detach().cpu().numpy().astype(np.int32),
            ),
        ),
        shape=(n_cells, n_cells),
    )
    A_csr.sum_duplicates()
    return A_csr


def _solve_csr_components(
    A_csr: csr_array,
    rhs: Float[torch.Tensor, " C *component_shape"],
    x0: Float[torch.Tensor, " C *component_shape"],
    *,
    atol: float,
    rtol: float,
    max_iter: int,
    norm_order: int | float,
) -> SolveResult:
    """Apply V-cycles with the same residual criterion as Krylov solvers.

    PyAMG's native tolerance uses ||b|| and an L2 norm. Check the true
    residual ourselves after each cycle to support configured norms and
    a relative tolerance based on the initial residual, including b=0.
    The hierarchy is built once and reused across cycles and components.
    """
    validate_component_shape(tuple(rhs.shape[1:]))
    if A_csr.shape != (rhs.shape[0], rhs.shape[0]):
        raise ValueError("AMG matrix must be square with one row per cell")
    require_shape(x0, tuple(rhs.shape), "AMG initial guess")
    b_np = rhs.detach().cpu().numpy()
    x0_np = x0.detach().cpu().numpy()
    x_out = np.zeros_like(b_np)
    ml = None
    stats = []

    for c in component_indices(tuple(rhs.shape[1:])):
        index = (slice(None), *c)
        b = b_np[index]
        x = np.asarray(x0_np[index], dtype=b_np.dtype).copy()
        initial = float(np.linalg.norm(b - A_csr @ x, ord=norm_order))
        threshold = max(atol, rtol * initial)
        final = initial
        iterations = 0
        while np.isfinite(final) and final >= threshold:
            if iterations >= max_iter:
                break
            if ml is None:
                ml = pyamg.smoothed_aggregation_solver(A_csr)
            x = ml.solve(b, x0=x, tol=0.0, maxiter=1)
            iterations += 1
            final = float(np.linalg.norm(b - A_csr @ x, ord=norm_order))
        x_out[index] = x
        stats.append(
            SolveStats(
                solver="pyamg",
                initial_residual=initial,
                final_residual=final,
                iterations=iterations,
                converged=bool(np.isfinite(final) and final < threshold),
            )
        )

    return SolveResult(
        solution=torch.from_numpy(x_out).to(device=rhs.device, dtype=rhs.dtype),
        stats=tuple(stats),
    )


class PyamgBridgeSolver(LinearSolver):
    """
    Bridge solver to PyAMG (algebraic multigrid).

    Intended for symmetric systems that benefit from strong multigrid
    preconditioning, such as pressure Poisson equations.
    """

    def __init__(self, config: SolverConfig):
        self.atol = config.tolerance
        self.rtol = config.rel_tolerance
        self.max_iter = config.max_iter
        self.norm_order = config.norm_type.to_norm_order()

    def _solve_primal(self, eq: Equation) -> SolveResult:
        A = eq.fv_matrix
        A_csr = _build_csr(
            A.diag, A.upper, A.lower, A.grid.owner, A.grid.neighbour
        )
        return _solve_csr_components(
            A_csr,
            A.source,
            eq.target.data,
            atol=self.atol,
            rtol=self.rtol,
            norm_order=self.norm_order,
            max_iter=self.max_iter,
        )

    def solve_transpose(
        self,
        A_T: FvMatrix,
        rhs: Float[torch.Tensor, " C *component_shape"],
    ) -> Float[torch.Tensor, " C *component_shape"]:
        A_csr = _build_csr(
            A_T.diag, A_T.upper, A_T.lower, A_T.grid.owner, A_T.grid.neighbour
        )
        return _solve_csr_components(
            A_csr,
            rhs,
            torch.zeros_like(rhs),
            atol=self.atol,
            rtol=self.rtol,
            norm_order=self.norm_order,
            max_iter=self.max_iter,
        ).solution

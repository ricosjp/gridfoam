from __future__ import annotations

import numpy as np
import pyamg
import torch
from jaxtyping import Float, Int
from scipy.sparse import csr_array

from gridfoam.core.equation import Equation
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.meta.config import SolverConfig
from gridfoam.solvers.base import LinearSolver, SolveResult, SolveStats


def _build_csr(
    diag: Float[torch.Tensor, " C 1"],
    upper: Float[torch.Tensor, " F 1"],
    lower: Float[torch.Tensor, " F 1"],
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
    vals = torch.cat(
        [diag.reshape(-1), upper.reshape(-1), lower.reshape(-1)], dim=0
    )

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
    rhs: Float[torch.Tensor, " C k"],
    x0: Float[torch.Tensor, " C k"],
    *,
    tol: float,
    max_iter: int,
) -> Float[torch.Tensor, " C k"]:
    """Solve each RHS column with PyAMG smoothed aggregation."""
    b_np = rhs.detach().cpu().numpy()
    x0_np = x0.detach().cpu().numpy()
    x_out = np.zeros_like(b_np)
    ml = pyamg.smoothed_aggregation_solver(A_csr)

    for c in range(b_np.shape[1]):
        x = np.asarray(x0_np[:, c], dtype=b_np.dtype).copy()
        x_out[:, c] = ml.solve(b_np[:, c], x0=x, tol=tol, maxiter=max_iter)

    return torch.from_numpy(x_out).to(device=rhs.device, dtype=rhs.dtype)


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

    def _solve_primal(self, eq: Equation) -> SolveResult:
        A = eq.fv_matrix
        A_csr = _build_csr(
            A.diag, A.upper, A.lower, A.grid.owner, A.grid.neighbour
        )
        solution = _solve_csr_components(
            A_csr,
            A.source,
            eq.target.data,
            tol=self.rtol,
            max_iter=self.max_iter,
        )
        return SolveResult(
            solution=solution,
            stats=(
                SolveStats(
                    solver="pyamg",
                    initial_residual=0.0,
                    final_residual=0.0,
                    iterations=0,
                    converged=True,
                ),
            ),
        )

    def solve_transpose(
        self,
        A_T: FvMatrix,
        rhs: Float[torch.Tensor, " C k"],
    ) -> Float[torch.Tensor, " C k"]:
        A_csr = _build_csr(
            A_T.diag, A_T.upper, A_T.lower, A_T.grid.owner, A_T.grid.neighbour
        )
        return _solve_csr_components(
            A_csr,
            rhs,
            torch.zeros_like(rhs),
            tol=self.rtol,
            max_iter=self.max_iter,
        )

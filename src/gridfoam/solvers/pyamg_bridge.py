from __future__ import annotations

import numpy as np
import pyamg
import torch
from jaxtyping import Float, Int
from scipy.sparse import csr_array

from gridfoam.core.equation import Equation
from gridfoam.meta.config import SolverConfig
from gridfoam.solvers.base import LinearSolver


class _PyamgSolveFunction(torch.autograd.Function):
    """
    Custom autograd function for PyAMG solve calls.

    Uses adjoint solves in backward pass to propagate analytical gradients
    through an external solver that is outside the native graph.
    """

    @staticmethod
    def _build_csr(
        diag: Float[torch.Tensor, " C 1"],
        upper: Float[torch.Tensor, " F 1"],
        lower: Float[torch.Tensor, " F 1"],
        owner: Int[torch.Tensor, " F"],
        neighbour: Int[torch.Tensor, " F"],
    ) -> csr_array:
        n_cells = int(diag.shape[0])

        rows = [
            torch.arange(n_cells, device=diag.device, dtype=torch.long),
            owner,
            neighbour,
        ]
        cols = [
            torch.arange(n_cells, device=diag.device, dtype=torch.long),
            neighbour,
            owner,
        ]
        vals = [diag.reshape(-1), upper.reshape(-1), lower.reshape(-1)]

        rows_np = torch.cat(rows, dim=0).detach().cpu().numpy().astype(np.int32)
        cols_np = torch.cat(cols, dim=0).detach().cpu().numpy().astype(np.int32)
        vals_np = torch.cat(vals, dim=0).detach().cpu().numpy()
        A_csr = csr_array(
            (vals_np, (rows_np, cols_np)),
            shape=(n_cells, n_cells),
        )
        A_csr.sum_duplicates()
        return A_csr

    @staticmethod
    def _solve_system(
        A_csr: csr_array,
        rhs: Float[np.ndarray, " C 1"],
        x0: Float[np.ndarray, " C 1"],
        tol: float,
        max_iter: int,
    ) -> Float[np.ndarray, " C 1"]:
        ml = pyamg.smoothed_aggregation_solver(A_csr)
        x = np.asarray(x0, dtype=rhs.dtype).copy()
        x = ml.solve(rhs, x0=x, tol=tol, maxiter=max_iter)
        return x

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        diag: Float[torch.Tensor, " C 1"],
        upper: Float[torch.Tensor, " F 1"],
        lower: Float[torch.Tensor, " F 1"],
        source: Float[torch.Tensor, " C k"],
        x0: Float[torch.Tensor, " C k"],
        owner: Int[torch.Tensor, " F"],
        neighbour: Int[torch.Tensor, " F"],
        rtol: float,
        max_iter: int,
    ) -> Float[torch.Tensor, " C k"]:
        A_csr = _PyamgSolveFunction._build_csr(
            diag=diag,
            upper=upper,
            lower=lower,
            owner=owner,
            neighbour=neighbour,
        )

        b_np = source.detach().cpu().numpy()
        x0_np = x0.detach().cpu().numpy()
        x_out = np.zeros_like(b_np)

        for c in range(b_np.shape[1]):
            x_out[:, c] = _PyamgSolveFunction._solve_system(
                A_csr=A_csr,
                rhs=b_np[:, c],
                x0=x0_np[:, c],
                tol=rtol,
                max_iter=max_iter,
            )

        x_tensor = torch.from_numpy(x_out).to(
            device=source.device,
            dtype=source.dtype,
        )

        ctx.rtol = rtol
        ctx.max_iter = max_iter
        ctx.save_for_backward(
            diag,
            upper,
            lower,
            owner,
            neighbour,
            x_tensor,
        )
        return x_tensor

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Float[torch.Tensor, " C k"],
    ) -> tuple[torch.Tensor | None, ...]:
        (
            diag,
            upper,
            lower,
            owner,
            neighbour,
            x,
        ) = ctx.saved_tensors

        A_csr = _PyamgSolveFunction._build_csr(
            diag=diag,
            upper=upper,
            lower=lower,
            owner=owner,
            neighbour=neighbour,
        )
        AT_csr = A_csr.transpose().tocsr()

        g_np = grad_output.detach().cpu().numpy()
        lambda_np = np.zeros_like(g_np)

        for c in range(g_np.shape[1]):
            lambda_np[:, c] = _PyamgSolveFunction._solve_system(
                A_csr=AT_csr,
                rhs=g_np[:, c],
                x0=np.zeros_like(g_np[:, c]),
                tol=ctx.rtol,
                max_iter=ctx.max_iter,
            )

        lambda_t = torch.from_numpy(lambda_np).to(
            device=grad_output.device,
            dtype=grad_output.dtype,
        )

        grad_diag = -(lambda_t * x).sum(dim=1, keepdim=True)
        grad_upper = -(lambda_t[owner] * x[neighbour]).sum(dim=1, keepdim=True)
        grad_lower = -(lambda_t[neighbour] * x[owner]).sum(dim=1, keepdim=True)
        grad_source = lambda_t

        grad_x0 = torch.zeros_like(x)

        return (
            grad_diag,
            grad_upper,
            grad_lower,
            grad_source,
            grad_x0,
            None,  # owner
            None,  # neighbour
            None,  # rtol
            None,  # max_iter
        )


class PyamgBridgeSolver(LinearSolver):
    """
    Bridge solver to PyAMG (algebraic multigrid).

    Intended for symmetric systems that benefit from strong multigrid
    preconditioning, such as pressure Poisson equations.
    """

    def __init__(self, config: SolverConfig):
        self.rtol = config.rel_tolerance
        self.max_iter = config.max_iter

    def solve(
        self,
        eq: Equation,
    ) -> Float[torch.Tensor, " C k"]:
        A = eq.lhs
        x = eq.target.data
        owner = A.grid.owner
        neighbour = A.grid.neighbour

        return _PyamgSolveFunction.apply(
            A.diag,
            A.upper,
            A.lower,
            A.source,
            x,
            owner,
            neighbour,
            self.rtol,
            self.max_iter,
        )

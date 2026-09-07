import logging

import torch
from jaxtyping import Float

from gridfoam.core.equation import Equation
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.meta.config import SolverConfig
from gridfoam.solvers.base import (
    LinearSolver,
    SolveResult,
    SolveStats,
    is_converged,
    residual_threshold,
)
from gridfoam.solvers.krylov import solve_components
from gridfoam.solvers.preconditioners import (
    Preconditioner,
)

logger = logging.getLogger(__name__)


class CGSolver(LinearSolver):
    """
    Conjugate Gradient (CG) solver.

    Iterative solver for symmetric positive-definite systems
    (for example, pressure equations).
    """

    def __init__(self, config: SolverConfig):
        self.solver_name = config.method.value
        self.precon_type = config.preconditioner
        self.atol = config.tolerance
        self.rtol = config.rel_tolerance
        self.max_iter = config.max_iter
        self.norm_order = config.norm_type.to_norm_order()
        self.log_interval = config.log_interval

    def _solve_primal(
        self,
        eq: Equation,
    ) -> SolveResult:
        A = eq.fv_matrix
        logger.info(
            "CG solve start eq=%s num_components=%d max_iter=%d",
            eq.name,
            A.num_components,
            self.max_iter,
        )
        solution, stats = solve_components(
            A,
            A.source,
            eq.target.data,
            precon_type=self.precon_type,
            solve_single=self._solve_single,
        )
        logger.info("CG solve end eq=%s", eq.name)
        return SolveResult(solution=solution, stats=stats)

    def solve_transpose(
        self,
        A_T: FvMatrix,
        rhs: Float[torch.Tensor, " C *component_shape"],
    ) -> Float[torch.Tensor, " C *component_shape"]:
        solution, _ = solve_components(
            A_T,
            rhs,
            torch.zeros_like(rhs),
            precon_type=self.precon_type,
            solve_single=self._solve_single,
        )
        return solution

    def _solve_single(
        self,
        A: FvMatrix,
        b: Float[torch.Tensor, " C"],
        x: Float[torch.Tensor, " C"],
        precon: Preconditioner,
    ) -> tuple[Float[torch.Tensor, " C"], SolveStats]:
        """
        CG loop for a single scalar component.

        Expects ``x`` with shape ``[C]``.
        """
        # Initial residual r = b - A x
        r = b - A.multiply(x)

        norm_r0: torch.Tensor = torch.linalg.vector_norm(
            r, dim=0, ord=self.norm_order
        )

        thresh = residual_threshold(self.atol, self.rtol, norm_r0)
        initial_residual = norm_r0.item()

        logger.debug(
            "CG init residual=%.3e threshold=%.3e",
            initial_residual,
            thresh.item(),
        )

        if is_converged(thresh, norm_r0):
            logger.debug(
                "CG converged at iteration=0 residual=%.3e",
                initial_residual,
            )
            return x, SolveStats(
                solver=self.solver_name,
                initial_residual=initial_residual,
                final_residual=initial_residual,
                iterations=0,
                converged=True,
            )

        z = precon.apply(r)
        p = z.clone()

        rz_old = torch.sum(r * z)
        converged = False
        final_residual = initial_residual
        iter_idx = 0

        for iter_idx in range(1, self.max_iter + 1):
            Ap = A.multiply(p)

            p_Ap = torch.sum(p * Ap)
            alpha = rz_old / p_Ap

            x = x + alpha * p
            r = r - alpha * Ap

            # Early stop as soon as convergence criterion is met.
            norm_r: torch.Tensor = torch.linalg.vector_norm(
                r, dim=0, ord=self.norm_order
            )
            final_residual = norm_r.item()
            if iter_idx % self.log_interval == 0:
                logger.debug(
                    "CG iter=%d residual=%.3e threshold=%.3e",
                    iter_idx,
                    final_residual,
                    thresh.item(),
                )
            if is_converged(thresh, norm_r):
                logger.debug(
                    "CG converged iter=%d residual=%.3e",
                    iter_idx,
                    final_residual,
                )
                converged = True
                break

            z = precon.apply(r)
            rz_new = torch.sum(r * z)

            beta = rz_new / rz_old
            p = z + p * beta

            rz_old = rz_new

        else:
            logger.warning(
                "CG reached max_iter=%d without convergence",
                self.max_iter,
            )

        return x, SolveStats(
            solver=self.solver_name,
            initial_residual=initial_residual,
            final_residual=final_residual,
            iterations=iter_idx,
            converged=converged,
        )

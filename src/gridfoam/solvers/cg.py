import logging

import torch
from jaxtyping import Float

from gridfoam.core.equation import Equation
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.meta.config import SolverConfig
from gridfoam.solvers.base import (
    LinearSolver,
    is_converged,
    residual_threshold,
)
from gridfoam.solvers.preconditioners import (
    Preconditioner,
    create_preconditioner,
)

logger = logging.getLogger(__name__)


class CGSolver(LinearSolver):
    """
    Conjugate Gradient (CG) solver.

    Iterative solver for symmetric positive-definite systems
    (for example, pressure equations).
    """

    def __init__(self, config: SolverConfig):
        self.precon_type = config.preconditioner
        self.atol = config.tolerance
        self.rtol = config.rel_tolerance
        self.max_iter = config.max_iter
        self.norm_order = config.norm_type.to_norm_order()
        self.log_interval = config.log_interval

    def solve(
        self,
        eq: Equation,
    ) -> Float[torch.Tensor, " C k"]:
        A = eq.lhs
        b = A.source
        x = eq.target.data
        k = A.num_components
        logger.info(
            "CG solve start eq=%s components=%d max_iter=%d",
            eq.name,
            k,
            self.max_iter,
        )

        # Preconditioner
        precon = create_preconditioner(self.precon_type, A)

        x_res = []
        for c in range(k):
            x_res.append(
                self._solve_single(A, b[:, c : c + 1], x[:, c : c + 1], precon)
            )

        # Concatenate solved components back to [C, k].
        x_out = torch.cat(x_res, dim=1)
        logger.info("CG solve end eq=%s", eq.name)
        return x_out

    def _solve_single(
        self,
        A: FvMatrix,
        b: Float[torch.Tensor, " C 1"],
        x: Float[torch.Tensor, " C 1"],
        precon: Preconditioner,
    ) -> Float[torch.Tensor, " C 1"]:
        """
        CG loop for a single scalar component.

        Expects ``x`` with shape ``[C, 1]``.
        """
        norm_b: torch.Tensor = torch.linalg.vector_norm(
            b, dim=0, ord=self.norm_order
        )

        # Initial residual r = b - A x
        r = b - A.multiply(x)

        norm_r0: torch.Tensor = torch.linalg.vector_norm(
            r, dim=0, ord=self.norm_order
        )

        ref_norm = torch.maximum(norm_b, norm_r0)
        thresh = residual_threshold(self.atol, self.rtol, ref_norm)

        logger.debug(
            "CG init residual=%.3e threshold=%.3e",
            norm_r0.item(),
            thresh.item(),
        )

        if is_converged(thresh, norm_r0):
            logger.debug(
                "CG converged at iteration=0 residual=%.3e",
                norm_r0.item(),
            )
            return x

        z = precon.apply(r)
        p = z.clone()

        rz_old = torch.sum(r * z)

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
            if iter_idx % self.log_interval == 0:
                logger.debug(
                    "CG iter=%d residual=%.3e threshold=%.3e",
                    iter_idx,
                    norm_r.item(),
                    thresh.item(),
                )
            if is_converged(thresh, norm_r):
                logger.debug(
                    "CG converged iter=%d residual=%.3e",
                    iter_idx,
                    norm_r.item(),
                )
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

        return x

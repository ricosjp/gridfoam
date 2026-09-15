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
    residual_threshold,
)
from gridfoam.solvers.krylov import solve_components
from gridfoam.solvers.preconditioners import (
    Preconditioner,
)

logger = logging.getLogger(__name__)


class BiCGSTABSolver(LinearSolver):
    """
    Biconjugate Gradient Stabilized (BiCGSTAB) solver.

    Iterative solver for non-symmetric systems, such as momentum
    equations with convection terms.
    """

    def __init__(self, config: SolverConfig):
        self.solver_name = config.method.value
        self.precon_type = config.preconditioner
        self.atol = config.tolerance
        self.rtol = config.rel_tolerance
        self.max_iter = config.max_iter
        self.norm_order = config.norm_type.to_norm_order()
        self.max_restart = config.max_restart
        self.log_interval = config.log_interval

    def _solve_primal(
        self,
        eq: Equation,
    ) -> SolveResult:
        A = eq.fv_matrix
        logger.info(
            "BiCGSTAB solve start eq=%s num_components=%d max_iter=%d",
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
        logger.info("BiCGSTAB solve end eq=%s", eq.name)
        return SolveResult(solution=solution, stats=stats)

    def solve_transpose(
        self,
        A_T: FvMatrix,
        rhs: Float[torch.Tensor, " C *component_shape"],
    ) -> Float[torch.Tensor, " C *component_shape"]:
        solution, stats = solve_components(
            A_T,
            rhs,
            torch.zeros_like(rhs),
            precon_type=self.precon_type,
            solve_single=self._solve_single,
        )
        self._check_result(
            SolveResult(solution, stats), A_T.field.name, "Transpose"
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
        BiCGSTAB loop for a single scalar component.

        Expects ``x`` with shape ``[C]``.
        """
        # Initial residual r = b - A x
        r = b - A.multiply(x)

        norm_r0: torch.Tensor = torch.linalg.vector_norm(
            r, dim=0, ord=self.norm_order
        )

        thresh = residual_threshold(self.atol, self.rtol, norm_r0).item()
        initial_residual = norm_r0.item()

        logger.debug(
            "BiCGSTAB init residual=%.3e threshold=%.3e",
            initial_residual,
            thresh,
        )

        if initial_residual < thresh:
            logger.debug(
                "BiCGSTAB converged at iteration=0 residual=%.3e",
                initial_residual,
            )
            return x, SolveStats(
                solver=self.solver_name,
                initial_residual=initial_residual,
                final_residual=initial_residual,
                iterations=0,
                converged=True,
            )

        restart_count = 0
        r_hat, p, rho_old = self._restart_state(r)
        converged = False
        final_residual = initial_residual
        iter_idx = 0

        for iter_idx in range(1, self.max_iter + 1):
            # Apply right preconditioner: y = M^{-1} p
            y = precon.apply(p)
            v = A.multiply(y)

            denom_alpha = torch.sum(r_hat * v)
            if torch.abs(denom_alpha) <= self._breakdown_tol(r_hat, v):
                if not self._can_restart(
                    restart_count,
                    iter_idx,
                    "alpha denominator near zero",
                ):
                    break
                restart_count += 1
                r_hat, p, rho_old = self._restart_state(r)
                continue
            alpha = rho_old / denom_alpha

            s = r - alpha * v

            # Early convergence check
            norm_s: torch.Tensor = torch.linalg.vector_norm(
                s, dim=0, ord=self.norm_order
            )
            final_residual = norm_s.item()
            if iter_idx % self.log_interval == 0:
                logger.debug(
                    "BiCGSTAB iter=%d residual_s=%.3e threshold=%.3e",
                    iter_idx,
                    final_residual,
                    thresh,
                )
            if final_residual < thresh:
                x = x + alpha * y
                logger.debug(
                    "BiCGSTAB converged via s iter=%d residual=%.3e",
                    iter_idx,
                    final_residual,
                )
                converged = True
                break

            # Apply right preconditioner: z = M^{-1} s
            z = precon.apply(s)
            t = A.multiply(z)

            denom_omega = torch.sum(t * t, dim=0)
            if torch.abs(denom_omega) <= self._breakdown_tol(t, t):
                if not self._can_restart(
                    restart_count,
                    iter_idx,
                    "omega denominator near zero",
                ):
                    break
                restart_count += 1
                r_hat, p, rho_old = self._restart_state(r)
                continue

            omega = torch.sum(t * s, dim=0) / denom_omega
            if torch.abs(omega) <= self._breakdown_tol():
                if not self._can_restart(
                    restart_count,
                    iter_idx,
                    "omega near zero",
                ):
                    break
                restart_count += 1
                r_hat, p, rho_old = self._restart_state(r)
                continue

            x = x + alpha * y + omega * z
            r = s - omega * t

            norm_r: torch.Tensor = torch.linalg.vector_norm(
                r, dim=0, ord=self.norm_order
            )
            final_residual = norm_r.item()
            if final_residual < thresh:
                logger.debug(
                    "BiCGSTAB converged iter=%d residual=%.3e",
                    iter_idx,
                    final_residual,
                )
                converged = True
                break

            rho_new = torch.sum(r_hat * r)
            if torch.abs(rho_new) <= self._breakdown_tol(r_hat, r):
                if not self._can_restart(
                    restart_count,
                    iter_idx,
                    "rho near zero",
                ):
                    break
                restart_count += 1
                r_hat, p, rho_old = self._restart_state(r)
                continue

            beta = (rho_new / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)
            rho_old = rho_new
        else:
            logger.warning(
                "BiCGSTAB reached max_iter=%d without convergence",
                self.max_iter,
            )

        return x, SolveStats(
            solver=self.solver_name,
            initial_residual=initial_residual,
            final_residual=final_residual,
            iterations=iter_idx,
            converged=converged,
        )

    def _can_restart(
        self,
        restart_count: int,
        iter_idx: int,
        reason: str,
    ) -> bool:
        """Log breakdown reason and check restart budget."""
        logger.warning("BiCGSTAB breakdown at iter=%d: %s", iter_idx, reason)
        return restart_count < self.max_restart

    def _restart_state(
        self,
        r: Float[torch.Tensor, " C"],
    ) -> tuple[
        Float[torch.Tensor, " C"],
        Float[torch.Tensor, " C"],
        Float[torch.Tensor, ""],
    ]:
        """Return restarted (r_hat, p, rho_old) state."""
        r_hat = r.clone()
        p = r.clone()
        rho_old = torch.sum(r * r)
        return r_hat, p, rho_old

    def _breakdown_tol(
        self,
        x: Float[torch.Tensor, " N"] | None = None,
        y: Float[torch.Tensor, " N"] | None = None,
        *,
        eps: float = 1e-6,
    ) -> float:
        """Return relative tolerance based on vector norms."""
        x_norm = 1.0 if x is None else torch.linalg.vector_norm(x, dim=0).item()
        y_norm = 1.0 if y is None else torch.linalg.vector_norm(y, dim=0).item()
        return eps * x_norm * y_norm

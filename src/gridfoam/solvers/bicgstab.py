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


class BiCGSTABSolver(LinearSolver):
    """
    Biconjugate Gradient Stabilized (BiCGSTAB) solver.

    Iterative solver for non-symmetric systems, such as momentum
    equations with convection terms.
    """

    def __init__(self, config: SolverConfig):
        self.precon_type = config.preconditioner
        self.atol = config.tolerance
        self.rtol = config.rel_tolerance
        self.max_iter = config.max_iter
        self.norm_order = config.norm_type.to_norm_order()
        self.max_restart = config.max_restart
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
            "BiCGSTAB solve start eq=%s components=%d max_iter=%d",
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
        logger.info("BiCGSTAB solve end eq=%s", eq.name)
        return x_out

    def _solve_single(
        self,
        A: FvMatrix,
        b: Float[torch.Tensor, " C 1"],
        x: Float[torch.Tensor, " C 1"],
        precon: Preconditioner,
    ) -> Float[torch.Tensor, " C 1"]:
        """
        BiCGSTAB loop for a single scalar component.

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
            "BiCGSTAB init residual=%.3e threshold=%.3e",
            norm_r0.item(),
            thresh.item(),
        )

        if is_converged(thresh, norm_r0):
            logger.debug(
                "BiCGSTAB converged at iteration=0 residual=%.3e",
                norm_r0.item(),
            )
            return x

        restart_count = 0
        r_hat, p, rho_old = self._restart_state(r)

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
            if iter_idx % self.log_interval == 0:
                logger.debug(
                    "BiCGSTAB iter=%d residual_s=%.3e threshold=%.3e",
                    iter_idx,
                    norm_s.item(),
                    thresh.item(),
                )
            if is_converged(thresh, norm_s):
                x = x + alpha * y
                logger.debug(
                    "BiCGSTAB converged via s iter=%d residual=%.3e",
                    iter_idx,
                    norm_s.item(),
                )
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
            if is_converged(thresh, norm_r):
                logger.debug(
                    "BiCGSTAB converged iter=%d residual=%.3e",
                    iter_idx,
                    norm_r.item(),
                )
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

        return x

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
        r: Float[torch.Tensor, " C 1"],
    ) -> tuple[
        Float[torch.Tensor, " C 1"],
        Float[torch.Tensor, " C 1"],
        Float[torch.Tensor, ""],
    ]:
        """Return restarted (r_hat, p, rho_old) state."""
        r_hat = r.clone()
        p = r.clone()
        rho_old = torch.sum(r * r)
        return r_hat, p, rho_old

    def _breakdown_tol(
        self,
        x: Float[torch.Tensor, " N 1"] | None = None,
        y: Float[torch.Tensor, " N 1"] | None = None,
        *,
        eps: float = 1e-6,
    ) -> float:
        """Return relative tolerance based on vector norms."""
        x_norm = 1.0 if x is None else torch.linalg.vector_norm(x, dim=0).item()
        y_norm = 1.0 if y is None else torch.linalg.vector_norm(y, dim=0).item()
        return eps * x_norm * y_norm

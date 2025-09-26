import torch
from jaxtyping import Float

from gridfoam._simulator._equation import Expr
from gridfoam._simulator._solver._interface import Solver
from gridfoam.cubion import PyOctreeLevel


class BiCGSTAB(Solver):
    def __init__(self, expr: Expr):
        self.expr = expr
        self.x0 = None
        # default configurations
        self.max_iter = 1000
        self.tol = 1e-6

    def configure(self, max_iter: int, tol: float) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        if self.x0 is None:
            xi = torch.zeros(octree_level.n_cells)
        else:
            xi = self.x0.clone()
        vi = torch.zeros(octree_level.n_cells)
        ri = xi - self.expr.matvec(octree_level, xi, dt, dx)
        r0 = ri.clone()
        rho_old, rho_new = 1.0, 1.0
        alpha, beta, omega = 1.0, 1.0, 1.0
        diag = self.expr.diag(octree_level, dt, dx)
        if torch.any(diag == 0.0):
            raise ValueError("diag is 0")
        pi = ri.clone()
        for i in range(self.max_iter):
            rho_new = r0.dot(ri)
            if i > 0:
                beta = (rho_new / rho_old) * (alpha / omega)
                pi = ri + beta * (pi - omega * vi)
            phat = pi / diag
            vi = self.expr.matvec(octree_level, phat, dt, dx)
            alpha = rho_new / r0.dot(vi)
            s = ri - alpha * vi
            shat = s / diag
            t = self.expr.matvec(octree_level, shat, dt, dx)
            omega = t.dot(s) / t.dot(t)

            if omega == 0.0:
                raise ValueError("omega is 0")
            xi += alpha * phat + omega * shat
            ri = s - omega * t

            # L_2 norm
            residual = torch.norm(ri, p=2)
            if residual < self.tol:
                print(
                    f"BiCGSTAB converged\
                        -- iterations: {i + 1}, residual: {residual}"
                )
                self.x0 = xi.clone()
                return xi
            rho_old = rho_new
        raise ValueError(
            f"BiCGSTAB did not converge\
                -- iterations: {self.max_iter}, residual: {residual}"
        )

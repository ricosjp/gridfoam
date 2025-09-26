import torch
from jaxtyping import Float

from gridfoam._simulator._equation import Expr
from gridfoam._simulator._solver._interface import Solver
from gridfoam.cubion import PyOctreeLevel


class CG(Solver):
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
        ri = xi - self.expr.matvec(octree_level, xi, dt, dx)
        diag = self.expr.diag(octree_level, dt, dx)
        if torch.any(diag == 0.0):
            raise ValueError("diag is 0")
        pi = ri / diag
        for i in range(self.max_iter):
            qi = self.expr.matvec(octree_level, pi, dt, dx)
            tmp = ri.dot(ri / diag)
            alpha = tmp / pi.dot(qi)
            xi += alpha * pi
            ri -= alpha * qi
            z = ri / diag
            beta = ri.dot(z) / tmp
            pi = z + beta * pi

            # L_inf norm
            residual = torch.max(torch.abs(ri))
            if residual < self.tol:
                print(
                    f"CG converged\
                        -- iterations: {i + 1}, residual: {residual}"
                )
                self.x0 = xi.clone()
                return xi
        raise ValueError(
            f"CG did not converge\
                -- iterations: {self.max_iter}, residual: {residual}"
        )

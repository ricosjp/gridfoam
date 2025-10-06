import torch
from jaxtyping import Float

from gridfoam._interface import IFVMTerm, ISolver
from gridfoam.cubion import PyOctreeLevel


class BiCGSTAB(ISolver):
    def __init__(self, expr: IFVMTerm):
        self.expr = expr
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
        xi = torch.zeros(octree_level.n_leaf_cells)
        vi = torch.zeros(octree_level.n_leaf_cells)
        ri = self.expr.rhs(octree_level, dt, dx) - self.expr.matvec(
            octree_level, xi, dt, dx
        )
        residual_0 = torch.norm(ri, p=2)
        if residual_0 < 1e-12:
            return xi
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
            t_dot_t = t.dot(t)
            if t_dot_t < 1e-12:
                omega = 0.0
            else:
                omega = t.dot(s) / t_dot_t
            xi += alpha * phat + omega * shat
            ri = s - omega * t

            # L_2 norm
            residual_1 = torch.norm(ri, p=2)
            rel_residual = residual_1 / residual_0
            print(f"rel_residual: {rel_residual}")
            if rel_residual < self.tol:
                print(
                    f"BiCGSTAB converged\
                        -- iterations: {i + 1}, rel_residual: {rel_residual}"
                )
                self.x0 = xi.clone()
                return xi
            rho_old = rho_new
        raise ValueError(
            f"BiCGSTAB did not converge\
                -- iterations: {self.max_iter}, rel_residual: {rel_residual}"
        )

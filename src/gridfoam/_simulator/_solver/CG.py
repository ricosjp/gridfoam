import torch
from jaxtyping import Float

from gridfoam._interface._solver import Solver
from gridfoam._simulator._scheme._term._expr import Expr
from gridfoam.cubion import PyOctreeLevel


class CG(Solver):
    """
    Conjugate Gradient (CG) solver for linear systems.

    This class implements the Conjugate Gradient method for solving
    linear systems of equations. It is particularly effective for
    symmetric positive definite matrices.
    """

    def __init__(self, expr: Expr):
        """
        Initialize the CG solver.

        Parameters
        ----------
        expr : Expr
            The expression representing the linear system to solve.
        """
        self.expr = expr
        self.x0 = None
        # default configurations
        self.max_iter = 1000
        self.tol = 1e-6

    def configure(self, max_iter: int, tol: float) -> None:
        """
        Configure the solver parameters.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.
        """
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Solve the linear system using the Conjugate Gradient method.

        Parameters
        ----------
        octree_level : PyOctreeLevel
            The octree level containing the grid data.
        dt : float
            Time step size.
        dx : Float[torch.Tensor, " 3"]
            Grid spacing in each direction.

        Returns
        -------
        torch.Tensor
            The solution vector.

        Raises
        ------
        ValueError
            If the diagonal contains zeros or if convergence is not achieved.
        """
        if self.x0 is None:
            xi = torch.zeros(octree_level.n_leaf_cells)
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

import torch
from jaxtyping import Float

from gridfoam.DNA._gridhandle import IGridHandle
from gridfoam.DNA.config import SolverChoice
from gridfoam.DNA.enum import FieldLayout, FieldRole, NormType
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.solver._interface import ILinearSolver


class BiCGSTAB(ILinearSolver):
    """
    Bi-Conjugate Gradient Stabilized (BiCGSTAB) solver for linear systems.

    This class implements the BiCGSTAB method for solving
    linear systems of equations. It is a variant of the BiCG method
    that is more stable and efficient.
    """

    def __init__(
        self,
        solver_choice: SolverChoice,
        eq_meta: EquationMeta,
    ) -> None:
        self._preconditioner = solver_choice.preconditioner
        self._tolerance = solver_choice.tolerance
        self._rel_tolerance = solver_choice.rel_tolerance
        self._max_iter = solver_choice.max_iter
        self._norm_type = solver_choice.norm_type
        self._eq_meta = eq_meta

        n_components = eq_meta.target_field.components
        dtype = eq_meta.target_field.dtype
        self._p_fm = FieldMeta(
            name=f"_BiCGSTAB.p(components={n_components}, dtype={dtype})",
            label="Search direction",
            description="Search direction for the BiCGSTAB solver",
            role=FieldRole.AUXILIARY,
            layout=FieldLayout.CELL,
            components=n_components,
            dtype=dtype,
            default_output=False,
        )
        self._s_fm = FieldMeta(
            name=f"_BiCGSTAB.s(components={n_components}, dtype={dtype})",
            label="Intermediate residual",
            description="Intermediate residual for the BiCGSTAB solver",
            role=FieldRole.AUXILIARY,
            layout=FieldLayout.CELL,
            components=n_components,
            dtype=dtype,
            default_output=False,
        )

    @property
    def required_fields(self) -> list[FieldMeta]:
        return [self._p_fm, self._s_fm]

    def solve(
        self,
        grid_handle: IGridHandle,
    ) -> None:
        """
        Solve the linear system using the BiCGSTAB method.

        Parameters
        ----------
        grid_handle : GridHandle
            Grid handle.
        """
        eq_name = self._eq_meta.name
        x_fm = self._eq_meta.target_field
        n_leaf_nodes = grid_handle.grid.n_leaf_nodes
        N = grid_handle.config.cube.interior_width
        device = grid_handle.config.cube.device
        shape = (n_leaf_nodes, x_fm.components, N, N, N)
        x = torch.zeros(shape, device=device, dtype=x_fm.dtype)
        r = torch.zeros(shape, device=device, dtype=x_fm.dtype)
        r0 = torch.zeros(shape, device=device, dtype=x_fm.dtype)
        p = torch.zeros(shape, device=device, dtype=x_fm.dtype)
        y = torch.zeros(shape, device=device, dtype=x_fm.dtype)
        z = torch.zeros(shape, device=device, dtype=x_fm.dtype)

        for i, (_, cube) in enumerate(grid_handle.iter_all_leaves()):
            field = cube.field
            xi = field.cells[x_fm.name]
            fvmatrix = field.fvmatrices[eq_name]
            x[i] = xi.interior[0]
            r[i] = fvmatrix.source.interior[0] - fvmatrix.apply(xi)
            r0[i] = r[i]
            p[i] = r[i]
            field.cells[self._p_fm.name].interior[0] = p[i]
        grid_handle.sync_all([self._p_fm])

        rnorm_0 = self._compute_norm(r0)
        if rnorm_0 < self._tolerance:
            print("no need to solve")
            return

        for it in range(self._max_iter):
            # Update solution
            for i, (_, cube) in enumerate(grid_handle.iter_all_leaves()):
                field = cube.field
                fvmatrix = field.fvmatrices[eq_name]
                pi = field.cells[self._p_fm.name]
                y[i] = fvmatrix.apply(pi)
            r0r = (r0 * r).sum()
            r0y = (r0 * y).sum()
            alpha = r0r / r0y if torch.abs(r0y) > 1e-12 else 0.0
            s = r - alpha * y
            for i, (_, cube) in enumerate(grid_handle.iter_all_leaves()):
                field = cube.field
                field.cells[self._s_fm.name].interior[0] = s[i]
            grid_handle.sync_all([self._s_fm])

            for i, (_, cube) in enumerate(grid_handle.iter_all_leaves()):
                field = cube.field
                fvmatrix = field.fvmatrices[eq_name]
                si = field.cells[self._s_fm.name]
                z[i] = fvmatrix.apply(si)
            zz = (z * z).sum()
            omega = (z * s).sum() / zz if torch.abs(zz) > 1e-12 else 0.0
            x += alpha * p + omega * s
            r = s - omega * z

            # Check convergence
            rnorm = self._compute_norm(r)
            print(
                f"BiCGSTAB iteration {it + 1}, \
                    residual: {rnorm:.6f}, rel_residual: {rnorm / rnorm_0:.6f}"
            )
            if (
                rnorm < self._tolerance
                or rnorm / rnorm_0 < self._rel_tolerance
                or omega == 0.0
            ):
                print(
                    f"BiCGSTAB converged\
                        -- iterations: {it + 1},\
                        residual: {rnorm:.6f},\
                        rel_residual: {rnorm / rnorm_0:.6f}"
                )
                for i, (_, cube) in enumerate(grid_handle.iter_all_leaves()):
                    field = cube.field
                    field.cells[x_fm.name].interior[0] = x[i]
                grid_handle.sync_all([x_fm])
                return

            # Update search direction
            beta = (alpha / omega) * ((r0 * r).sum() / r0r)
            p = r + beta * (p - omega * y)
            for i, (_, cube) in enumerate(grid_handle.iter_all_leaves()):
                field = cube.field
                field.cells[self._p_fm.name].interior[0] = p[i]
            grid_handle.sync_all([self._p_fm])

        raise ValueError(
            f"BiCGSTAB did not converge\
                -- iterations: {self._max_iter},\
                residual: {rnorm:.6f},\
                rel_residual: {rnorm / rnorm_0:.6f}"
        )

    def _compute_norm(self, x: Float[torch.Tensor, "..."]) -> float:
        match self._norm_type:
            case NormType.L_inf:
                return torch.max(torch.abs(x)).item()
            case NormType.L_2:
                return torch.norm(x, p=2).item()

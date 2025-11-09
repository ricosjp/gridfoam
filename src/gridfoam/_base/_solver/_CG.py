from collections.abc import Iterator

import torch
from jaxtyping import Float

from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._field._grid import PyOctreeNode
from gridfoam._base._field._handle import FieldHandle
from gridfoam._base._field._registry import FieldRegistry
from gridfoam._base._interface._fvm_term import IFVMTerm
from gridfoam._base._interface._solver import ISolver
from gridfoam.utils.enums import FieldLayout, FieldRole, Namespace, NormType


class CG(ISolver):
    """
    Conjugate Gradient solver for linear systems.
    """

    def __init__(
        self,
        preconditioner: str | None,
        tolerance: float,
        rel_tolerance: float,
        max_iter: int,
        field_registry: FieldRegistry,
        norm_type: NormType = NormType.L_inf,
    ) -> None:
        self.preconditioner = preconditioner
        self.tolerance = tolerance
        self.rel_tolerance = rel_tolerance
        self.max_iter = max_iter
        self.norm_type = norm_type

        self._p_fd = field_registry.declare(
            name="search_direction",
            channels=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
            role=FieldRole.AUXILIARY,
            layout=FieldLayout.CELL,
            namespace=Namespace.SOLVER,
        )

    def solve(
        self,
        target_fd: FieldDescriptor,
        fvm_system: IFVMTerm,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> None:
        """
        Solve the linear system using the Conjugate Gradient method.

        Parameters
        ----------
        target_fd : FieldDescriptor
            The descriptor of the target field to solve for.
        fvm_system : IFVMTerm
            The FVM system to solve.
        dt : float
            Time step size.
        dx : Float[torch.Tensor, " 3"]
            Grid spacing in each direction.

        Raises
        ------
        ValueError
            If convergence is not achieved.
        """
        C = target_fd.channels
        N = FieldHandle._N

        n_leaf_nodes = FieldHandle._grid.n_leaf_nodes
        shape = (n_leaf_nodes, C, N, N, N)
        x = torch.zeros(shape, device=target_fd.device, dtype=target_fd.dtype)
        r = torch.zeros(shape, device=target_fd.device, dtype=target_fd.dtype)
        p = torch.zeros(shape, device=target_fd.device, dtype=target_fd.dtype)
        y = torch.zeros(shape, device=target_fd.device, dtype=target_fd.dtype)

        for i, (cube, depth) in enumerate(self._iter_field_with_sync([self._p_fd])):
            dx_local = FieldHandle.get_dx_at(depth)
            x[i] = cube.old.cells[target_fd.canonical_name].interior
            r[i] = fvm_system.source(cube, dt, dx_local) - fvm_system.matvec(
                cube, target_fd.canonical_name, dt, dx_local
            )
            p[i] = r[i]
            cube.old.cells[self._p_fd.canonical_name].interior = p[i]

        rnorm_0 = self._compute_norm(r)
        if rnorm_0 < self.tolerance:
            print("no need to solve")
            return
        for k in range(self.max_iter):
            # Update solution
            for i, (cube, depth) in enumerate(self._iter_field()):
                dx_local = FieldHandle.get_dx_at(depth)
                y[i] = fvm_system.matvec(
                    cube, self._p_fd.canonical_name, dt, dx_local
                )
            rr = (r * r).sum()
            py = (p * y).sum()
            alpha = rr / py if torch.abs(py) > 1e-12 else 0.0
            x += alpha * p
            r -= alpha * y

            # Check convergence
            rnorm = self._compute_norm(r)
            print(
                f"CG iteration {k + 1}, residual: {rnorm}, rel_residual: {rnorm / rnorm_0}"
            )
            if rnorm < self.tolerance or rnorm / rnorm_0 < self.rel_tolerance:
                print(
                    f"CG converged\
                        -- iterations: {k + 1},\
                        residual: {rnorm},\
                        rel_residual: {rnorm / rnorm_0}"
                )
                for i, (cube, _) in enumerate(self._iter_field_with_sync([target_fd])):
                    cube.old.cells[target_fd.canonical_name].interior = x[i]
                return

            # Update search direction
            beta = (r * r).sum() / rr
            p = r + beta * p
            for i, (cube, _) in enumerate(self._iter_field_with_sync([self._p_fd])):
                cube.old.cells[self._p_fd.canonical_name].interior = p[i]

        raise ValueError(
            f"CG did not converge\
                -- iterations: {self.max_iter},\
                residual: {rnorm},\
                rel_residual: {rnorm / rnorm_0}"
        )

    def _compute_norm(self, x: Float[torch.Tensor, "..."]) -> float:
        match self.norm_type:
            case NormType.L_inf:
                return torch.max(torch.abs(x)).item()
            case NormType.L_2:
                return torch.norm(x, p=2).item()

    def _iter_field_with_sync(self, fd_list: list[FieldDescriptor]) -> Iterator[tuple[PyOctreeNode, int]]:
        yield from self._iter_field_with_sync_at_depth(0, fd_list)

    def _iter_field(self) -> Iterator[tuple[PyOctreeNode, int]]:
        yield from self._iter_field_at_depth(0)

    def _iter_field_with_sync_at_depth(self, depth: int, fd_list: list[FieldDescriptor]) -> Iterator[tuple[PyOctreeNode, int]]:
        if depth == FieldHandle._grid.max_depth - 1:
            for cube in FieldHandle.iter_leaf_cubes_of(FieldHandle.get_octree_level_at(depth)):
                yield cube, depth
            FieldHandle.sync_halo_at_depth(depth, fd_list)
            return
        FieldHandle.sync_ghost_from_parent_at_depth(depth + 1, fd_list)
        FieldHandle.sync_halo_at_depth(depth + 1, fd_list)
        yield from self._iter_field_with_sync_at_depth(depth + 1, fd_list)
        FieldHandle.sync_ghost_from_children_at_depth(depth, fd_list)
        FieldHandle.sync_halo_at_depth(depth, fd_list)
        for cube in FieldHandle.iter_leaf_cubes_of(FieldHandle.get_octree_level_at(depth)):
            yield cube, depth
        FieldHandle.sync_halo_at_depth(depth, fd_list)

    def _iter_field_at_depth(self, depth: int) -> Iterator[tuple[PyOctreeNode, int]]:
        if depth == FieldHandle._grid.max_depth - 1:
            for cube in FieldHandle.iter_leaf_cubes_of(FieldHandle.get_octree_level_at(depth)):
                yield cube, depth
            return
        yield from self._iter_field_at_depth(depth + 1)
        for cube in FieldHandle.iter_leaf_cubes_of(FieldHandle.get_octree_level_at(depth)):
            yield cube, depth

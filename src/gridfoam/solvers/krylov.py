"""Shared helpers for Krylov linear solvers."""

from __future__ import annotations

from collections.abc import Callable

import torch
from jaxtyping import Float

from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.shapes import (
    component_indices,
    require_shape,
    stack_components,
)
from gridfoam.meta.enums import PreconditionerType
from gridfoam.solvers.base import SolveStats
from gridfoam.solvers.preconditioners import (
    Preconditioner,
    create_preconditioner,
)

KrylovSolveSingleFn = Callable[
    [
        FvMatrix,
        Float[torch.Tensor, " C"],
        Float[torch.Tensor, " C"],
        Preconditioner,
    ],
    tuple[Float[torch.Tensor, " C"], SolveStats],
]


def solve_components(
    A: FvMatrix,
    rhs: Float[torch.Tensor, " C *component_shape"],
    x0: Float[torch.Tensor, " C *component_shape"],
    *,
    precon_type: PreconditionerType,
    solve_single: KrylovSolveSingleFn,
) -> tuple[
    Float[torch.Tensor, " C *component_shape"],
    tuple[SolveStats, ...],
]:
    """
    Solve ``A x = rhs`` one physical component at a time.

    Parameters
    ----------
    A : FvMatrix
        Linear operator. Each scalar component shares the same LDU
        coefficients.
    rhs : torch.Tensor
        Right-hand side with shape ``[C, *component_shape]``.
    x0 : torch.Tensor
        Initial guess with shape ``[C, *component_shape]``.
    precon_type : PreconditionerType
        Preconditioner reused across components.
    solve_single : callable
        Scalar Krylov solve ``(A, b, x0, precon) -> (x, stats)``.

    Returns
    -------
    solution : torch.Tensor
        Solution with shape ``[C, *component_shape]``.
    stats : tuple[SolveStats, ...]
        Per-component solver statistics in physical-component order.
    """
    shape = (A.grid.num_cells, *A.field.component_shape)
    require_shape(rhs, shape, "rhs")
    require_shape(x0, shape, "initial guess")
    precon = create_preconditioner(precon_type, A)
    solutions: list[torch.Tensor] = []
    stats_list: list[SolveStats] = []
    for c in component_indices(A.field.component_shape):
        index = (slice(None), *c)
        x_c, stats = solve_single(A, rhs[index], x0[index], precon)
        solutions.append(x_c)
        stats_list.append(stats)
    return stack_components(solutions, A.field.component_shape), tuple(
        stats_list
    )

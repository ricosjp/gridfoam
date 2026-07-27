"""Shared helpers for Krylov linear solvers."""

from __future__ import annotations

from collections.abc import Callable

import torch
from jaxtyping import Float

from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.meta.enums import PreconditionerType
from gridfoam.solvers.base import SolveStats
from gridfoam.solvers.preconditioners import (
    Preconditioner,
    create_preconditioner,
)

KrylovSolveSingleFn = Callable[
    [
        FvMatrix,
        Float[torch.Tensor, " C 1"],
        Float[torch.Tensor, " C 1"],
        Preconditioner,
    ],
    tuple[Float[torch.Tensor, " C 1"], SolveStats],
]


def solve_transpose_components(
    A_T: FvMatrix,
    rhs: Float[torch.Tensor, " C k"],
    *,
    precon_type: PreconditionerType,
    solve_single: KrylovSolveSingleFn,
) -> Float[torch.Tensor, " C k"]:
    """
    Solve ``A_T y = rhs`` one component at a time via ``solve_single``.

    Parameters
    ----------
    A_T : FvMatrix
        Transpose view of the primal matrix.
    rhs : torch.Tensor
        Right-hand side with shape ``[C, k]``.
    precon_type : PreconditionerType
        Preconditioner for the transpose system.
    solve_single : callable
        Scalar Krylov solve ``(A, b, x0, precon) -> (x, stats)``.

    Returns
    -------
    torch.Tensor
        Solution with shape ``[C, k]``.
    """
    precon = create_preconditioner(precon_type, A_T)
    cols: list[torch.Tensor] = []
    for c in range(rhs.shape[1]):
        g_c = rhs[:, c : c + 1]
        y_c, _ = solve_single(A_T, g_c, torch.zeros_like(g_c), precon)
        cols.append(y_c)
    return torch.cat(cols, dim=1)

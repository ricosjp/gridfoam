"""Implicit adjoint attachment for linear solvers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.solvers.adjoint.ldu_grads import assemble_ldu_grads

SolveTransposeFn = Callable[
    [FvMatrix, Float[torch.Tensor, " C *component_shape"]],
    Float[torch.Tensor, " C *component_shape"],
]


class _AttachImplicitAdjoint(torch.autograd.Function):
    """
    Autograd hook for ``x = A^{-1} b``.

    Forward returns the detached primal solution. Backward solves
    ``A^T \\lambda = \\bar{x}`` and assembles LDU / source gradients.
    """

    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401
        diag: Float[torch.Tensor, " C"],
        upper: Float[torch.Tensor, " F"],
        lower: Float[torch.Tensor, " F"],
        source: Float[torch.Tensor, " C *component_shape"],
        sol: Float[torch.Tensor, " C *component_shape"],
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        field: CellField,
        solve_transpose: SolveTransposeFn,
    ) -> Float[torch.Tensor, " C *component_shape"]:
        ctx.save_for_backward(diag, upper, lower, owner, neighbour, sol)
        ctx.field = field
        ctx.solve_transpose = solve_transpose
        return sol

    @staticmethod
    def backward(
        ctx: Any,  # noqa: ANN401
        *grad_outputs: Float[torch.Tensor, " C *component_shape"],
    ) -> tuple[torch.Tensor | None, ...]:
        grad_output = grad_outputs[0]
        diag, upper, lower, owner, neighbour, x = ctx.saved_tensors

        A_mat = FvMatrix(ctx.field)
        A_mat.diag = diag.detach()
        A_mat.upper = upper.detach()
        A_mat.lower = lower.detach()
        A_mat.source = torch.zeros_like(grad_output)

        lambda_t = ctx.solve_transpose(
            A_mat.as_transpose(), grad_output.detach()
        )
        grad_diag, grad_upper, grad_lower, grad_source = assemble_ldu_grads(
            lambda_t, x, owner, neighbour
        )
        return (
            grad_diag,
            grad_upper,
            grad_lower,
            grad_source,
            None,  # sol
            None,  # owner
            None,  # neighbour
            None,  # field
            None,  # solve_transpose
        )


def attach_implicit_adjoint(
    A: FvMatrix,
    solution: Float[torch.Tensor, " C *component_shape"],
    *,
    solve_transpose: SolveTransposeFn,
) -> Float[torch.Tensor, " C *component_shape"]:
    """
    Attach an implicit adjoint to a detached linear-solve solution.

    Forward returns ``solution`` unchanged. Backward solves
    ``A^T \\lambda = \\bar{x}`` with ``solve_transpose``, then assembles
    LDU / source gradients via :func:`assemble_ldu_grads`.

    Parameters
    ----------
    A : FvMatrix
        Primal matrix whose coefficients participate in the autograd graph.
    solution : torch.Tensor
        Detached primal solution with shape ``[C, *component_shape]``.
    solve_transpose : callable
        Solves ``A_T y = rhs``, returning shape ``[C, *component_shape]``.
    """
    return cast(
        torch.Tensor,
        _AttachImplicitAdjoint.apply(
            A.diag,
            A.upper,
            A.lower,
            A.source,
            solution,
            A.grid.owner,
            A.grid.neighbour,
            A.field,
            solve_transpose,
        ),
    )

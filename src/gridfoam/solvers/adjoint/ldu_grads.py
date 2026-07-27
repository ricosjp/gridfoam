"""Shared LDU gradient assembly for implicit linear-solve adjoints."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int


def assemble_ldu_grads(
    lambda_t: Float[torch.Tensor, " C k"],
    x: Float[torch.Tensor, " C k"],
    owner: Int[torch.Tensor, " F"],
    neighbour: Int[torch.Tensor, " F"],
) -> tuple[
    Float[torch.Tensor, " C 1"],
    Float[torch.Tensor, " F 1"],
    Float[torch.Tensor, " F 1"],
    Float[torch.Tensor, " C k"],
]:
    """
    Assemble gradients of ``x = A^{-1} b`` w.r.t. LDU coefficients and ``b``.

    For cotangent ``lambda_t = A^{-T} \\bar{x}`` and primal solution ``x``,

    - ``grad_diag_i = -sum_k lambda_{i,k} x_{i,k}``
    - ``grad_upper_f = -sum_k lambda_{owner,k} x_{neighbour,k}``
    - ``grad_lower_f = -sum_k lambda_{neighbour,k} x_{owner,k}``
    - ``grad_source = lambda_t``

    Parameters
    ----------
    lambda_t : torch.Tensor
        Adjoint variable with shape ``[C, k]``.
    x : torch.Tensor
        Primal solution with shape ``[C, k]``.
    owner : torch.Tensor
        Owner cell indices per internal face.
    neighbour : torch.Tensor
        Neighbour cell indices per internal face.

    Returns
    -------
    tuple of torch.Tensor
        ``(grad_diag, grad_upper, grad_lower, grad_source)``.
    """
    return (
        -(lambda_t * x).sum(dim=1, keepdim=True),
        -(lambda_t[owner] * x[neighbour]).sum(dim=1, keepdim=True),
        -(lambda_t[neighbour] * x[owner]).sum(dim=1, keepdim=True),
        lambda_t,
    )

"""Optional finite-difference check for the inlet-temperature objective."""

import logging
import math
from collections.abc import Callable

import torch


def check_gradient_fd(
    evaluate: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    inlet_temperature: torch.Tensor,
    *,
    eps: float = 1e-3,
) -> None:
    """Compare autograd with central differences at the initial inlet value."""
    inlet = inlet_temperature.detach().clone().requires_grad_()
    with torch.no_grad():
        loss_center, _ = evaluate(inlet)
        loss_plus, _ = evaluate(inlet + eps)
        loss_minus, _ = evaluate(inlet - eps)
        fd_grad = ((loss_plus - loss_minus) / (2.0 * eps)).item()

    loss, _ = evaluate(inlet)
    (gradient,) = torch.autograd.grad(loss, inlet)
    autograd_grad = gradient.item()
    rel_err = abs(autograd_grad - fd_grad) / max(abs(fd_grad), 1e-12)
    logging.getLogger("gridfoam.examples.optimize").info(
        "grad check: loss=%.4e fd=%.4e autograd=%.4e rel_err=%.2e",
        loss_center.item(),
        fd_grad,
        autograd_grad,
        rel_err,
    )
    if not math.isclose(autograd_grad, fd_grad, rel_tol=1e-3, abs_tol=1e-6):
        raise RuntimeError(
            f"Gradient check failed: autograd={autograd_grad:.6e}, "
            f"FD={fd_grad:.6e}. Unrolled Krylov gradients can be unstable "
            "even when the primal solve converges; use --grad-mode adjoint."
        )

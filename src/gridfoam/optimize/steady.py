"""Converged fixed-point solves with an implicit, matrix-free backward."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol, cast, runtime_checkable

import torch

from gridfoam.core.state import TensorState


@runtime_checkable
class StepMap(Protocol):
    def step(self, state: TensorState, design: TensorState) -> TensorState: ...


@dataclass(frozen=True)
class SteadyOptions:
    """Separate primal and adjoint convergence budgets and tolerances."""

    max_steps: int = 1000
    primal_atol: float = 1e-9
    primal_rtol: float = 1e-8
    max_adjoint_steps: int = 400
    adjoint_atol: float = 1e-10
    adjoint_rtol: float = 1e-8
    restart: int = 40

    def __post_init__(self) -> None:
        if min(self.max_steps, self.max_adjoint_steps, self.restart) < 1:
            raise ValueError("Iteration budgets and restart must be positive")
        for value in (
            self.primal_atol,
            self.primal_rtol,
            self.adjoint_atol,
            self.adjoint_rtol,
        ):
            if not isfinite(value) or value < 0:
                raise ValueError("Tolerances must be finite and nonnegative")
        if (
            self.primal_atol + self.primal_rtol == 0
            or self.adjoint_atol + self.adjoint_rtol == 0
        ):
            raise ValueError("Each solve needs a positive tolerance")


class ConvergenceError(RuntimeError):
    """A primal or adjoint iteration failed its residual criterion."""


def _flat(state: TensorState) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in state.values()])


def _unflat(vector: torch.Tensor, layout: TensorState) -> TensorState:
    blocks = vector.split([value.numel() for value in layout.values()])
    return layout.from_tuple(
        tuple(
            block.reshape_as(value)
            for block, value in zip(blocks, layout.values(), strict=True)
        )
    )


def _finite(state: TensorState) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in state.values())


def _primal(
    mapping: StepMap,
    initial: TensorState,
    design: TensorState,
    options: SteadyOptions,
) -> TensorState:
    state = initial.checkpoint()
    with torch.no_grad():
        for _ in range(options.max_steps):
            next_state = mapping.step(state, design)
            state.validate_layout(next_state)
            if not _finite(next_state):
                raise ConvergenceError(
                    "Primal produced non-finite state values"
                )
            # Require each physical block to converge, avoiding domination by
            # a large auxiliary coefficient or a field with many components.
            converged = all(
                float(torch.linalg.vector_norm(next_state[key] - value))
                <= options.primal_atol
                + options.primal_rtol * float(torch.linalg.vector_norm(value))
                for key, value in state.items()
            )
            if converged:
                # G(state)-state was actually checked at this returned state.
                return state
            state = next_state.checkpoint()
    raise ConvergenceError(
        f"Primal did not converge in {options.max_steps} steps"
    )


def _gmres(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    options: SteadyOptions,
) -> torch.Tensor:
    """Restarted GMRES; accept only the actual unpreconditioned residual."""
    solution = torch.zeros_like(rhs)
    threshold = options.adjoint_atol + options.adjoint_rtol * float(
        torch.linalg.vector_norm(rhs)
    )
    residual = rhs.clone()
    norm = float(torch.linalg.vector_norm(residual))
    if norm <= threshold:
        return solution
    used = 0
    while used < options.max_adjoint_steps:
        beta = torch.linalg.vector_norm(residual)
        basis = [residual / beta]
        width = min(
            options.restart, options.max_adjoint_steps - used, rhs.numel()
        )
        hessenberg = rhs.new_zeros((width + 1, width))
        target = rhs.new_zeros(width + 1)
        target[0] = beta
        candidate = solution
        for j in range(width):
            vector = matvec(basis[j])
            if not bool(torch.isfinite(vector).all()):
                raise ConvergenceError(
                    "Adjoint produced a non-finite matrix product"
                )
            # Reorthogonalize to keep the small least-squares problem stable.
            for _ in range(2):
                for i in range(j + 1):
                    coefficient = torch.dot(basis[i], vector)
                    hessenberg[i, j] += coefficient
                    vector = vector - coefficient * basis[i]
            length = torch.linalg.vector_norm(vector)
            hessenberg[j + 1, j] = length
            coefficients = torch.linalg.lstsq(
                hessenberg[: j + 2, : j + 1], target[: j + 2]
            ).solution
            candidate = (
                solution + torch.stack(basis[: j + 1], dim=1) @ coefficients
            )
            residual = rhs - matvec(candidate)
            used += 1
            norm = float(torch.linalg.vector_norm(residual))
            if not isfinite(norm):
                raise ConvergenceError("Adjoint residual is non-finite")
            if norm <= threshold:
                return candidate
            if float(length) <= torch.finfo(rhs.dtype).eps:
                break
            basis.append(vector / length)
        solution = candidate
    raise ConvergenceError(
        f"Adjoint did not converge in {used} iterations; "
        f"residual={norm:.3e}, tolerance={threshold:.3e}"
    )


def _vjp(
    output: TensorState,
    inputs: tuple[torch.Tensor, ...],
    cotangent: TensorState,
) -> tuple[torch.Tensor, ...]:
    pairs = [
        (value, cotangent[key])
        for key, value in output.items()
        if value.requires_grad
    ]
    if not pairs:
        return tuple(torch.zeros_like(value) for value in inputs)
    grads = torch.autograd.grad(
        tuple(value for value, _ in pairs),
        inputs,
        grad_outputs=tuple(weight for _, weight in pairs),
        allow_unused=True,
        retain_graph=True,
    )
    return tuple(
        torch.zeros_like(value) if grad is None else grad
        for value, grad in zip(inputs, grads, strict=True)
    )


class _ImplicitSteady(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401
        mapping: StepMap,
        initial: TensorState,
        design_layout: TensorState,
        options: SteadyOptions,
        *parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        design = design_layout.from_tuple(parameters)
        state = _primal(mapping, initial, design, options)
        ctx.mapping = mapping
        # Keep layout placeholders off the output/input graphs. Actual tensors
        # belong in save_for_backward for lifetime and mutation checks.
        ctx.state_layout = state.zeros_like()
        ctx.design_layout = design_layout.zeros_like()
        ctx.options = options
        ctx.save_for_backward(*state.as_tuple(), *parameters)
        ctx.set_materialize_grads(False)
        return state.as_tuple()

    @staticmethod
    def backward(
        ctx: Any,  # noqa: ANN401
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Steady solves currently support first derivatives only"
            )
        n_state = len(ctx.state_layout)
        saved = ctx.saved_tensors
        with torch.enable_grad():
            state = (
                ctx.state_layout.from_tuple(saved[:n_state])
                .checkpoint()
                .requires_grad_()
            )
            design = (
                ctx.design_layout.from_tuple(saved[n_state:])
                .checkpoint()
                .requires_grad_()
            )
            output = ctx.mapping.step(state, design)
        cotangent = state.from_tuple(
            tuple(
                torch.zeros_like(value) if weight is None else weight
                for value, weight in zip(
                    state.values(), grad_outputs, strict=True
                )
            )
        )
        rhs = _flat(cotangent)
        if not bool(torch.isfinite(rhs).all()):
            raise ConvergenceError("Non-finite upstream gradient")

        def matvec(vector: torch.Tensor) -> torch.Tensor:
            jt = state.from_tuple(
                _vjp(output, state.as_tuple(), _unflat(vector, state))
            )
            return vector - _flat(jt)

        adjoint = _unflat(_gmres(matvec, rhs, ctx.options), state)
        sensitivity = _vjp(output, design.as_tuple(), adjoint)
        return (None, None, None, None, *sensitivity)


def steady_solve(
    mapping: StepMap,
    initial: TensorState,
    design: TensorState,
    *,
    options: SteadyOptions | None = None,
) -> TensorState:
    """Solve a fixed point and attach ``(I - dG/dq)^T`` adjoints in backward.

    Objectives are constructed from the returned tensors. Explicit objective
    dependence on design variables is handled by ordinary PyTorch autograd.
    The initial guess is numerical input, not a differentiable trajectory.
    All state blocks must share floating dtype/device; design blocks may have
    different shapes. Unconverged solves raise ``ConvergenceError``.
    """
    options = SteadyOptions() if options is None else options
    if torch.is_inference_mode_enabled():
        with torch.inference_mode(False), torch.no_grad():
            return steady_solve(
                mapping,
                initial.checkpoint(),
                design.checkpoint(),
                options=options,
            )
    if not initial or not design:
        raise ValueError("State and design mappings must be nonempty")
    first = next(iter(initial.values()))
    if any(
        not value.is_floating_point()
        or value.dtype != first.dtype
        or value.device != first.device
        for value in initial.values()
    ):
        raise ValueError("State blocks must share a floating dtype and device")
    if any(not value.is_floating_point() for value in design.values()):
        raise ValueError("Design blocks must be floating tensors")
    if not _finite(initial) or not _finite(design):
        raise ValueError("Initial state and design must be finite")
    values = cast(
        tuple[torch.Tensor, ...],
        _ImplicitSteady.apply(
            mapping, initial, design, options, *design.as_tuple()
        ),
    )
    return initial.from_tuple(values)

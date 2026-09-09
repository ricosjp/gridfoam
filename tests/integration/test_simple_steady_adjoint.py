"""Real SIMPLE fixed-point sensitivities and repeated inlet optimization."""

from pathlib import Path

import torch
from tests.integration.test_simple_step_map import make_step_map

from gridfoam.core.state import TensorState
from gridfoam.optimize.steady import SteadyOptions, steady_solve


def test_simple_steady_gradient_matches_finite_difference_and_unroll(
    tmp_path: Path,
) -> None:
    """SIMPLE fixed-point gradients match finite differences and unrolling."""
    mapping = make_step_map(tmp_path)
    direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    options = SteadyOptions(primal_atol=1e-11, primal_rtol=1e-10)
    speed = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)

    def objective(value: torch.Tensor) -> torch.Tensor:
        state = steady_solve(
            mapping,
            mapping.initial_state,
            TensorState({"inlet_velocity": value * direction}),
            options=options,
        )
        return (
            state["U"][:, 0].square().mean() + 0.01 * state["p"].square().mean()
        )

    grad = torch.autograd.grad(objective(speed), speed)[0]
    with torch.no_grad():
        fd = (objective(speed + 1e-4) - objective(speed - 1e-4)) / 2e-4
    torch.testing.assert_close(grad, fd, rtol=5e-5, atol=1e-6)
    state = mapping.initial_state
    design = TensorState({"inlet_velocity": speed * direction})
    for _ in range(100):
        state = mapping.step(state, design)
    unrolled = torch.autograd.grad(
        state["U"][:, 0].square().mean() + 0.01 * state["p"].square().mean(),
        speed,
    )[0]
    torch.testing.assert_close(grad, unrolled, rtol=5e-5, atol=1e-6)


def test_multiple_inlet_optimizer_updates_reduce_loss(tmp_path: Path) -> None:
    """Repeated inlet updates reduce loss after an inference evaluation."""
    mapping = make_step_map(tmp_path)
    direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    speed = torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float64))
    optimizer = torch.optim.SGD([speed], lr=0.2)
    # Trainer validation uses inference_mode before returning to training.
    with torch.inference_mode():
        evaluation = steady_solve(
            mapping,
            mapping.initial_state,
            TensorState({"inlet_velocity": speed * direction}),
        )
    assert not evaluation["U"].requires_grad
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        state = steady_solve(
            mapping,
            mapping.initial_state,
            TensorState({"inlet_velocity": speed * direction}),
        )
        loss = (state["U"][:, 0].mean() - 1.2).square()
        loss.backward()
        assert speed.grad is not None and torch.isfinite(speed.grad)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.1

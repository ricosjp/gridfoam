"""Check upstream gradients, convergence failures and graph isolation."""

import pytest
import torch

from gridfoam.core.state import TensorState
from gridfoam.optimize.steady import (
    ConvergenceError,
    SteadyOptions,
    steady_solve,
)


class CoupledMap:
    """Small linear fixed-point map with two state and design blocks."""

    def step(self, state: TensorState, design: TensorState) -> TensorState:
        """Apply one coupled fixed-point iteration."""
        x, y = state["x"], state["y"]
        return TensorState(
            {
                "x": 0.3 * x + 0.1 * y + design["a"],
                "y": 0.2 * x + 0.4 * y + design["b"],
            }
        )


def test_arbitrary_cotangents_and_multiple_design_blocks() -> None:
    """Implicit gradients match an exact solve for a general scalar loss."""
    initial = TensorState(
        {
            "x": torch.zeros((), dtype=torch.float64),
            "y": torch.zeros((), dtype=torch.float64),
        }
    )
    a = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    options = SteadyOptions(primal_atol=1e-12, primal_rtol=1e-12)
    result = steady_solve(
        CoupledMap(), initial, TensorState({"a": a, "b": b}), options=options
    )
    # An explicit design contribution is accumulated by ordinary autograd.
    loss = 2 * result["x"] - 3 * result["y"] + a.square()
    actual = torch.autograd.grad(loss, (a, b))
    matrix = a.new_tensor([[0.7, -0.1], [-0.2, 0.6]])
    exact = torch.linalg.solve(matrix, torch.stack((a, b)))
    expected = torch.autograd.grad(
        2 * exact[0] - 3 * exact[1] + a.square(), (a, b)
    )
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)


def test_two_pending_forwards_have_independent_backwards() -> None:
    """Two pending solves retain independent backward replay state."""
    initial = TensorState(
        {
            "x": torch.zeros((), dtype=torch.float64),
            "y": torch.zeros((), dtype=torch.float64),
        }
    )
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    first = steady_solve(CoupledMap(), initial, TensorState({"a": a, "b": b}))
    second = steady_solve(
        CoupledMap(), initial, TensorState({"a": 2 * a, "b": b})
    )
    gradient = torch.autograd.grad(first["x"] + second["y"], a)[0]
    torch.testing.assert_close(gradient, a.new_tensor(2.5))


def test_unconverged_primal_and_adjoint_raise() -> None:
    """Primal and adjoint iteration limits raise distinct failures."""
    initial = TensorState(
        {
            "x": torch.zeros((), dtype=torch.float64),
            "y": torch.zeros((), dtype=torch.float64),
        }
    )
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    design = TensorState({"a": a, "b": 2 * a})
    with pytest.raises(ConvergenceError, match="Primal"):
        steady_solve(
            CoupledMap(), initial, design, options=SteadyOptions(max_steps=1)
        )
    result = steady_solve(
        CoupledMap(),
        initial,
        design,
        options=SteadyOptions(max_adjoint_steps=1),
    )
    with pytest.raises(ConvergenceError, match="Adjoint"):
        result["x"].backward()

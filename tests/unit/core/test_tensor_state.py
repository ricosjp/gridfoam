"""What ``TensorState`` guarantees for named design blocks.

Graphs
    Heterogeneous blocks (inlet scalars and ``nn.Parameter``s) keep autograd
    links through ``clone``. ``checkpoint`` copies storage and drops the graph.

Layout
    Addition, VJP rebuild, and ``validate_layout`` reject reordered keys,
    a different block count, or a mismatched shape, dtype, or device.

Integers
    Discrete selections cannot be marked ``requires_grad``.
"""

import pytest
import torch

from gridfoam.core.state import TensorState


def test_named_design_blocks_keep_nn_parameter_gradients() -> None:
    """Inlet values and layer parameters remain distinct autograd leaves."""
    layer = torch.nn.Linear(2, 3, dtype=torch.float64)
    inlet = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    design = TensorState({"inlet": inlet, **dict(layer.named_parameters())})
    copy = design.clone()
    loss = copy["inlet"] * copy["weight"].square().sum() + copy["bias"].sum()
    loss.backward()
    torch.testing.assert_close(inlet.grad, layer.weight.detach().square().sum())
    torch.testing.assert_close(
        layer.weight.grad, 2 * inlet.detach() * layer.weight.detach()
    )
    torch.testing.assert_close(layer.bias.grad, torch.ones_like(layer.bias))


def test_checkpoint_storage_is_independent_and_has_no_graph() -> None:
    """A primal snapshot must survive later in-place writes to the source."""
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    state = TensorState({"T": source * 2})
    saved = state.checkpoint()
    assert not saved["T"].requires_grad
    with torch.no_grad():
        state["T"].add_(10)
    torch.testing.assert_close(saved["T"], torch.tensor([2.0, 4.0]))
    restored = saved.clone()
    restored["T"].zero_()
    torch.testing.assert_close(saved["T"], torch.tensor([2.0, 4.0]))


def test_vector_operations_reject_reordered_or_missing_blocks() -> None:
    """Key order and block count define the vector space."""
    state = TensorState({"U": torch.ones(4, 3), "T": torch.ones(4)})
    with pytest.raises(ValueError, match="keys or order"):
        _ = state + TensorState({"T": state["T"], "U": state["U"]})
    with pytest.raises(ValueError, match="block count"):
        state.from_tuple((state["U"],))
    assert (state + state.scale(-1.0)).norm() == 0.0


@pytest.mark.parametrize(
    "replacement",
    [
        torch.ones(1, 3),
        torch.ones(4, 3, dtype=torch.float64),
        torch.ones(4, 3, device="meta"),
    ],
    ids=["shape", "dtype", "device"],
)
def test_validate_layout_rejects_incompatible_blocks(
    replacement: torch.Tensor,
) -> None:
    """``+`` and ``from_tuple`` share this check; test the shared helper."""
    state = TensorState({"U": torch.ones(4, 3)})
    with pytest.raises(ValueError, match="layout"):
        state.validate_layout(TensorState({"U": replacement}))


def test_integer_selection_is_not_a_differentiable_state() -> None:
    """Connectivity and donor indices stay outside ``requires_grad_``."""
    state = TensorState(
        {"x": torch.ones(1), "indices": torch.zeros(1, dtype=torch.long)}
    )
    with pytest.raises(ValueError, match="floating-point"):
        state.requires_grad_()
    assert not state["x"].requires_grad

"""Rank-generic physical-axis helpers match the explicit tensor operations."""

import pytest
import torch

from gridfoam.core.shapes import (
    broadcast_entity,
    sum_physical,
)


def test_broadcast_entity_aligns_coefficients_without_copying() -> None:
    """
    Entity coefficients gain singleton physical axes while retaining
    storage and values.
    """
    coeff = torch.arange(8, dtype=torch.float32)
    for shape in ((), (3,), (3, 3)):
        values = torch.empty((8, *shape))
        aligned = broadcast_entity(coeff, values)
        assert aligned.shape == (8,) + (1,) * len(shape)
        assert aligned.data_ptr() == coeff.data_ptr()
        torch.testing.assert_close(aligned.reshape(-1), coeff)
    assert coeff.shape == (8,)


@pytest.mark.parametrize("component_shape", [(), (3,), (3, 3)])
@pytest.mark.parametrize("coeff_shape", [(4, 1), (1,)])
def test_broadcast_entity_rejects_incompatible_entity_coefficients(
    component_shape: tuple[int, ...], coeff_shape: tuple[int, ...]
) -> None:
    """
    Broadcasting rejects a wrong entity count or an already expanded
    coefficient array.
    """
    coeff = torch.ones(coeff_shape)
    value = torch.ones((4, *component_shape))
    with pytest.raises(ValueError, match="broadcast_entity requires"):
        broadcast_entity(coeff, value)


def test_broadcast_entity_and_sum_physical() -> None:
    """
    Mask broadcasting and tensor reduction retain the entity axis; scalars
    stay unchanged.
    """
    n = 4
    mask = torch.tensor([True, False, True, False])
    tensor = torch.arange(n * 9, dtype=torch.float32).reshape(n, 3, 3)
    aligned = broadcast_entity(mask, tensor)
    assert aligned.shape == (n, 1, 1)
    torch.testing.assert_close(sum_physical(tensor), tensor.sum(dim=(1, 2)))
    torch.testing.assert_close(sum_physical(tensor[:, 0, 0]), tensor[:, 0, 0])

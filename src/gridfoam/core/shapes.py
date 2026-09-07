"""Physical arrays use ``(entity, *component_shape)`` throughout gridfoam.

Scalar entity coefficients are one-dimensional. Align them with
``broadcast_entity`` before writing physical formulas with ordinary arithmetic.
For a known single spatial axis, use ``coeff[:, None]``: (N,) -> (N, 1).
Use explicit ``einsum`` indices where they clarify rank-generic tensor
contractions.
"""

from collections.abc import Iterator
from itertools import product
from math import prod

import torch


def validate_component_shape(shape: object) -> tuple[int, ...]:
    """Validate the spatial axes of a physical tensor (currently 3-D)."""
    if not isinstance(shape, tuple) or any(
        type(d) is not int or d != 3 for d in shape
    ):
        raise ValueError(
            "component_shape must be () or a tuple of spatial dimensions (3), "
            f"got {shape!r}"
        )
    return shape


def require_shape(
    value: torch.Tensor, shape: tuple[int, ...], name: str
) -> None:
    """Reject incompatible arrays at storage and API boundaries."""
    if tuple(value.shape) != shape:
        raise ValueError(
            f"{name}: shape {tuple(value.shape)}, expected {shape}"
        )


def broadcast_entity(entity: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape an entity scalar or mask ``(N,)`` to ``(N, 1, ..., 1)``.

    The trailing singleton axes match ``like.ndim - 1`` so ``entity``
    broadcasts against ``(N, *component_shape)``.
    """
    if entity.ndim != 1 or like.ndim < 1 or entity.shape[0] != like.shape[0]:
        raise ValueError(
            "broadcast_entity requires (N,) and (N, *component_shape)"
        )
    if like.ndim == 1:
        return entity
    return entity.reshape((entity.shape[0],) + (1,) * (like.ndim - 1))


def sum_physical(value: torch.Tensor) -> torch.Tensor:
    """Sum every physical axis, leaving the entity axis ``(N,)``."""
    axes = tuple(range(1, value.ndim))
    return value.sum(dim=axes) if axes else value


def component_indices(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Iterate physical component indices; a scalar has one index, ()."""
    return product(*(range(d) for d in shape))


def stack_components(
    values: list[torch.Tensor], shape: tuple[int, ...]
) -> torch.Tensor:
    """Assemble scalar solutions in physical component order."""
    if len(values) != prod(shape):
        raise ValueError("component count does not match component_shape")
    if not shape:
        return values[0]
    return torch.stack(values, dim=-1).reshape((values[0].shape[0],) + shape)

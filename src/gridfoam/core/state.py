"""Named tensor blocks for states, design inputs and their cotangents."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Self

import torch


class TensorState(Mapping[str, torch.Tensor]):
    """An ordered, read-only mapping that preserves tensor autograd links.

    Blocks may have different physical shapes (velocity, temperature, model
    parameters, ...). Keys and shapes define the vector space; reconstructing
    it must not silently drop, reorder or broadcast blocks. The tensors
    themselves remain mutable: use :meth:`clone` for storage isolation and
    :meth:`checkpoint` for an independent snapshot without an autograd graph.
    """

    def __init__(self, values: Mapping[str, torch.Tensor]):
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        """Return blocks in insertion order for autograd VJPs."""
        return tuple(self.values())

    def from_tuple(self, values: Sequence[torch.Tensor]) -> Self:
        """Rebuild with the same layout, preserving the supplied graph."""
        if len(values) != len(self):
            raise ValueError("Tensor state block count differs")
        blocks = dict(zip(self, values, strict=True))
        self.validate_layout(blocks)
        return type(self)(blocks)

    def validate_layout(self, values: Mapping[str, torch.Tensor]) -> None:
        """Require identical keys, order, shapes, dtypes and devices.

        Validate metadata only; no tensors are copied or states constructed.
        """
        if tuple(self) != tuple(values):
            raise ValueError("Tensor state keys or order differ")
        for key, reference in self.items():
            value = values[key]
            if (
                reference.shape != value.shape
                or reference.dtype != value.dtype
                or reference.device != value.device
            ):
                raise ValueError(f"Tensor state layout differs for {key!r}")

    def _map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        return type(self)({key: fn(value) for key, value in self.items()})

    def clone(self) -> Self:
        """Copy storage while preserving autograd links."""
        return self._map(torch.clone)

    def detach(self) -> Self:
        """Detach blocks; storage is still shared."""
        return self._map(torch.Tensor.detach)

    def checkpoint(self) -> Self:
        """Copy storage and discard the graph, suitable for primal replay."""
        return self._map(lambda value: value.detach().clone())

    def requires_grad_(self) -> Self:
        """Enable gradients on floating-point blocks in place."""
        if any(not value.is_floating_point() for value in self.values()):
            raise ValueError(
                "Only floating-point state blocks support gradients"
            )
        for value in self.values():
            value.requires_grad_(True)
        return self

    def zeros_like(self) -> Self:
        """Return a zero cotangent with this layout."""
        return self._map(torch.zeros_like)

    def __add__(self, other: Self) -> Self:
        self.validate_layout(other)
        return type(self)({key: self[key] + other[key] for key in self})

    def scale(self, alpha: float) -> Self:
        return self._map(lambda value: value * alpha)

    def norm(self) -> float:
        """Euclidean norm for diagnostics, with no autograd graph."""
        return (
            sum(
                float(torch.sum(value.detach().double().square()).item())
                for value in self.values()
            )
            ** 0.5
        )


def validate_history(
    current: torch.Tensor,
    old: torch.Tensor,
    older: torch.Tensor | None,
    previous_dt: float | None,
) -> None:
    """Validate a history replacement before any field buffers are changed."""
    for value in (old, older):
        if value is not None and (
            value.shape != current.shape
            or value.dtype != current.dtype
            or value.device != current.device
        ):
            raise ValueError(
                "History shape, dtype and device must match the field"
            )
    if (older is None) != (previous_dt is None):
        raise ValueError(
            "Second old level and previous_dt must be supplied together"
        )
    if previous_dt is not None and not (0.0 < previous_dt < float("inf")):
        raise ValueError("previous_dt must be finite and positive")

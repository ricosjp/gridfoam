"""Fixed donor interpolation and its exact transpose for GCIBM prototypes.

Donor selection belongs to geometry search. Weights are continuous inputs and
retain their autograd graph. This operator does not select fluid/ghost cells,
impose boundary values or alter the production LDU matrix/solver path.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InterpolationStencil:
    """GCIBM interpolation foothold with an explicit transpose.

    Donor search and production matrix assembly are not implemented here.
    Missing donors must use a valid index with zero weight, not ``-1``.
    """

    indices: torch.Tensor
    weights: torch.Tensor
    n_cells: int

    def __post_init__(self) -> None:
        if self.indices.ndim != 2 or self.indices.dtype != torch.long:
            raise ValueError(
                "Stencil indices must be a two-dimensional long tensor"
            )
        if (
            self.indices.shape != self.weights.shape
            or not self.weights.is_floating_point()
        ):
            raise ValueError(
                "Stencil weights must be floating point and match indices"
            )
        if self.indices.device != self.weights.device:
            raise ValueError("Stencil indices and weights must share a device")
        if self.n_cells < 1 or self.indices.shape[1] < 1:
            raise ValueError(
                "Stencil requires cells and at least one donor column"
            )
        if torch.any((self.indices < 0) | (self.indices >= self.n_cells)):
            raise ValueError("Stencil donor index is out of bounds")
        if not torch.all(torch.isfinite(self.weights)):
            raise ValueError("Stencil weights must be finite")

    def _validate_values(self, values: torch.Tensor, n_rows: int) -> None:
        if values.ndim < 1 or values.shape[0] != n_rows:
            raise ValueError("Stencil input has the wrong number of rows")
        if (
            values.dtype != self.weights.dtype
            or values.device != self.weights.device
        ):
            raise ValueError("Stencil input must match weight dtype and device")

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Interpolate arbitrary physical components on the trailing axes."""
        self._validate_values(values, self.n_cells)
        weights = self.weights.reshape(
            *self.weights.shape, *((1,) * (values.ndim - 1))
        )
        return (weights * values[self.indices]).sum(dim=1)

    def transpose_apply(self, values: torch.Tensor) -> torch.Tensor:
        """Scatter image-point cotangents, accumulating every repeated donor."""
        self._validate_values(values, self.indices.shape[0])
        weights = self.weights.reshape(
            *self.weights.shape, *((1,) * (values.ndim - 1))
        )
        contributions = (weights * values[:, None]).flatten(0, 1)
        result = values.new_zeros((self.n_cells, *values.shape[1:]))
        return result.index_add(0, self.indices.flatten(), contributions)

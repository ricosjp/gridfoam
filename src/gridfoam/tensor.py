from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch.types import Number


class GridTensor(torch.Tensor):
    def __new__(
        cls,
        data: Number | list[Number] | np.ndarray | torch.Tensor,
        coord_type: Literal["global", "local", "cell"] | None = "global",
        requires_grad: bool = False,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ):
        """
        Generate new GridTensor instance

        Parameters
        ----------
        data : Number | list | np.ndarray | torch.Tensor
            data to store in tensor
        coord_type : {"global", "local", "cell"} | None, optional
            coordinate type of tensor. default is "global".
        requires_grad : bool, optional
            whether to compute gradient for this tensor. default is False.
        *args, **kwargs :
            additional arguments to pass to `torch.as_tensor`
            like `device`, `dtype`, etc.

        Returns
        -------
        GridTensor
            new GridTensor instance
        """
        tensor = torch.as_tensor(data, *args, **kwargs)
        if tensor.grad_fn is None:
            tensor = tensor.requires_grad_(requires_grad)

        instance = torch.Tensor._make_subclass(
            cls, tensor, tensor.requires_grad
        )
        instance._coord_type = coord_type
        return instance

    def __init__(
        self,
        data: Number | list[Number] | np.ndarray | torch.Tensor,
        coord_type: Literal["global", "local", "cell"] | None = "global",
        requires_grad: bool = False,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ):
        pass

    def __repr__(self):
        return f"{super().__repr__()} (coord_type: {self.coord_type})"

    def __getitem__(self, indices: Any) -> GridTensor:
        result = super().__getitem__(indices)
        return GridTensor(result, self.coord_type)

    def _check_if_available_operation(self, other: Number | GridTensor):
        if isinstance(other, Number):
            return
        if not isinstance(other, GridTensor):
            raise TypeError(
                "Unsupported operand type(s) for operation: "
                f"'{type(self)}' and '{type(other)}'"
            )
        if self.coord_type != other.coord_type:
            raise TypeError(
                "Cannot operate on tensors with different coord_types: "
                f"'{self.coord_type}' and '{other.coord_type}'"
            )

    def __add__(self, other: Number | GridTensor) -> GridTensor:
        self._check_if_available_operation(other)
        result = super().__add__(other)
        return GridTensor(result, self.coord_type)

    def __sub__(self, other: Number | GridTensor) -> GridTensor:
        self._check_if_available_operation(other)
        result = super().__sub__(other)
        return GridTensor(result, self.coord_type)

    def __mul__(self, other: Number) -> GridTensor:
        if isinstance(other, Number):
            result = super().__mul__(other)
            return GridTensor(result, self.coord_type)
        return NotImplemented

    def __rmul__(self, other: Number) -> GridTensor:
        return self.__mul__(other)

    def __eq__(self, other: Number | GridTensor) -> GridTensor:
        self._check_if_available_operation(other)
        result = super().__eq__(other)
        return GridTensor(result, self.coord_type)

    def __lt__(self, other: Number | GridTensor) -> GridTensor:
        self._check_if_available_operation(other)
        result = super().__lt__(other)
        return GridTensor(result, self.coord_type)

    def __le__(self, other: Number | GridTensor) -> GridTensor:
        self._check_if_available_operation(other)
        result = super().__le__(other)
        return GridTensor(result, self.coord_type)

    def __gt__(self, other: Number | GridTensor) -> GridTensor:
        self._check_if_available_operation(other)
        result = super().__gt__(other)
        return GridTensor(result, self.coord_type)

    def __ge__(self, other: Number | GridTensor) -> GridTensor:
        self._check_if_available_operation(other)
        result = super().__ge__(other)
        return GridTensor(result, self.coord_type)

    @property
    def coord_type(self) -> Literal["global", "local", "cell"] | None:
        return getattr(self, "_coord_type", None)

    @coord_type.setter
    def coord_type(self, value: Literal["global", "local", "cell"] | None):
        if value not in ["global", "local", "cell", None]:
            raise ValueError(
                "coord_type must be one of: 'global', 'local', 'cell', None"
            )
        self._coord_type = value

    def clone(
        self, *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ) -> GridTensor:
        tensor = super().clone(*args, **kwargs)
        return GridTensor(tensor, self.coord_type)

    def to(
        self, *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ) -> GridTensor:
        tensor = super().to(*args, **kwargs)
        return GridTensor(
            tensor, self.coord_type, requires_grad=tensor.requires_grad
        )

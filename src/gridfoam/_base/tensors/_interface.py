from __future__ import annotations

import abc
from collections.abc import Callable, Sequence

import numpy as np
import torch

from gridfoam.utils.enums import CoordinateType


class IGridTensor(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def coord_type(self) -> CoordinateType | None: ...

    @property
    @abc.abstractmethod
    def shape(self) -> torch.Size: ...

    @property
    @abc.abstractmethod
    def ndim(self) -> int: ...

    @property
    @abc.abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...

    @abc.abstractmethod
    def numpy(self) -> np.ndarray: ...

    @abc.abstractmethod
    def tensor(self) -> torch.Tensor: ...

    @abc.abstractmethod
    def size(self) -> torch.Size: ...

    @abc.abstractmethod
    def numel(self) -> int: ...

    @abc.abstractmethod
    def reshape(self, shape: Sequence[int]) -> IGridTensor: ...

    @abc.abstractmethod
    def to(
        self,
        device: str,
        non_blocking: bool = False,
        dtype: torch.dtype | None = None,
    ) -> IGridTensor: ...

    @abc.abstractmethod
    def detach(self) -> IGridTensor: ...

    @abc.abstractmethod
    def backward(self) -> None: ...

    @abc.abstractmethod
    def clone(self) -> IGridTensor: ...

    @abc.abstractmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: list[type],
        args: tuple,
        kwargs: dict | None = None,
    ): ...

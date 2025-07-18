from __future__ import annotations

import abc

import torch
from jaxtyping import Float


class IAABB(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def space_dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def center(self) -> Float[torch.Tensor, " space_dim"]: ...

    @property
    @abc.abstractmethod
    def halfwidth(self) -> Float[torch.Tensor, " space_dim"]: ...

    @property
    @abc.abstractmethod
    def width(self) -> Float[torch.Tensor, " space_dim"]: ...

    @property
    @abc.abstractmethod
    def min(self) -> Float[torch.Tensor, " space_dim"]: ...

    @property
    @abc.abstractmethod
    def max(self) -> Float[torch.Tensor, " space_dim"]: ...

    @abc.abstractmethod
    def contains(self, pt: Float[torch.Tensor, " space_dim"]) -> bool: ...

    @abc.abstractmethod
    def scale(self, factor: float) -> None: ...

    @abc.abstractmethod
    def split(self) -> list[IAABB]: ...

    @abc.abstractmethod
    def merge(self, other: IAABB) -> IAABB: ...

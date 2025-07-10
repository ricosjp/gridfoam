from __future__ import annotations

import abc

import torch
from jaxtyping import Float


class IPlane(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def space_dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def normal(self) -> Float[torch.Tensor, " space_dim"]: ...

    @property
    @abc.abstractmethod
    def distance(self) -> float: ...

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(slots=True)
class IASTNode(abc.ABC):
    @abc.abstractmethod
    def __add__(self, other: IASTNode | None) -> IASTNode:
        pass

    @abc.abstractmethod
    def __sub__(self, other: IASTNode | None) -> IASTNode:
        pass

    @property
    @abc.abstractmethod
    def key(self) -> str:
        pass

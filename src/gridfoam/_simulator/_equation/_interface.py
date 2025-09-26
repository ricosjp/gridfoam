import abc

import torch
from jaxtyping import Float

from gridfoam.cubion import PyOctreeLevel


class FVMTerm(abc.ABC):
    @abc.abstractmethod
    def matvec(
        self,
        octree_level: PyOctreeLevel,
        x: torch.Tensor,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def diag(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def rhs(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        pass

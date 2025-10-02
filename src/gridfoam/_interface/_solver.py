import abc

import torch
from jaxtyping import Float

from gridfoam.cubion import PyOctreeLevel


class Solver(abc.ABC):
    @abc.abstractmethod
    def configure(self, max_iter: int, tol: float) -> None:
        pass

    @abc.abstractmethod
    def solve(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        pass

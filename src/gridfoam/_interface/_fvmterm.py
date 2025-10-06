import abc

import torch
from jaxtyping import Float

from gridfoam.cubion import PyOctreeLevel


class IFVMTerm(abc.ABC):
    @abc.abstractmethod
    def matvec(
        self,
        octree_level: PyOctreeLevel,
        x: torch.Tensor,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Apply the derivative operator to a vector.

        Parameters
        ----------
        octree_level : PyOctreeLevel
            The octree level containing the grid data.
        x : torch.Tensor
            Input vector to apply the operator to.
        dt : float
            Time step size.
        dx : Float[torch.Tensor, " 3"]
            Grid spacing in each direction.

        Returns
        -------
        torch.Tensor
            Result of applying the derivative operator.
        """
        pass

    @abc.abstractmethod
    def diag(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Get the diagonal elements of the derivative operator.

        Parameters
        ----------
        octree_level : PyOctreeLevel
            The octree level containing the grid data.
        dt : float
            Time step size.
        dx : Float[torch.Tensor, " 3"]
            Grid spacing in each direction.

        Returns
        -------
        torch.Tensor
            Diagonal elements of the derivative operator.
        """
        pass

    @abc.abstractmethod
    def rhs(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Compute the right-hand side vector for the derivative term.

        Parameters
        ----------
        octree_level : PyOctreeLevel
            The octree level containing the grid data.
        dt : float
            Time step size.
        dx : Float[torch.Tensor, " 3"]
            Grid spacing in each direction.

        Returns
        -------
        torch.Tensor
            Right-hand side vector for the derivative term.
        """
        pass

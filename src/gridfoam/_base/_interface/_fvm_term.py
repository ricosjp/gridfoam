import abc

import torch
from jaxtyping import Float

from gridfoam._base._field._grid import PyOctreeNode as Cube


class IFVMTerm(abc.ABC):
    @abc.abstractmethod
    def matvec(
        self,
        cube: Cube,
        target_name: str,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "C N N N"]:
        """
        Apply the implicit term(A) to the input field(x).

        Parameters
        ----------
        cube : Cube
            The grid cube.
        target_name : str
            Canonical name of the field to apply the implicit term(A) to.
        dt : float
            Time step size.
        dx : Float[torch.Tensor, " 3"]
            Grid spacing in each direction.

        Returns
        -------
        Float[torch.Tensor, "C N N N"]
            Result of applying the implicit term(A) to the input field(x).
        """
        pass

    @abc.abstractmethod
    def source(
        self, cube: Cube, dt: float, dx: Float[torch.Tensor, " 3"]
    ) -> Float[torch.Tensor, "C N N N"]:
        """
        Compute the explicit term(b).

        Parameters
        ----------
        cube : Cube
            The grid cube.
        dt : float
            Time step size.
        dx : Float[torch.Tensor, " 3"]
            Grid spacing in each direction.

        Returns
        -------
        Float[torch.Tensor, "C N N N"]
            Explicit term(b).
        """
        pass

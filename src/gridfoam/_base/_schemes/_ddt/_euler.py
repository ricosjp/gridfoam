import torch
from jaxtyping import Float

from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._field._grid import PyOctreeNode as Cube
from gridfoam._base._interface._fvm_term import IFVMTerm
from gridfoam._base._schemes._expr import FVMSchemeExpr
from gridfoam.utils.enums import DiscretizationMode


class EulerDdtScheme(IFVMTerm):
    """
    Time derivative term for finite volume method.

    This class represents the time derivative term (∂/∂t) in the finite
    volume method. It implements the temporal discretization for a given
    field variable.
    """

    def __init__(self, target_fd: FieldDescriptor, mode: DiscretizationMode):
        """
        Initialize the time derivative term.

        Parameters
        ----------
        target_fd : FieldDescriptor
            Field descriptor of the field to be integrated over time.
        mode : DiscretizationMode
            Mode of the discretization.
        """
        self._target_name = target_fd.canonical_name
        self._mode = mode


    def __add__(self, other: IFVMTerm) -> IFVMTerm:
        """
        Add another FVM term to this term.

        Parameters
        ----------
        other : IFVMTerm
            Another FVM term to add.

        Returns
        -------
        IFVMTerm
            An expression representing the sum of the two terms.
        """
        return FVMSchemeExpr(self, other, "+")

    def __sub__(self, other: IFVMTerm) -> IFVMTerm:
        """
        Subtract another FVM term from this term.

        Parameters
        ----------
        other : IFVMTerm
            Another FVM term to subtract.

        Returns
        -------
        IFVMTerm
            An expression representing the difference of the two terms.
        """
        return FVMSchemeExpr(self, other, "-")

    def matvec(
        self,
        cube: Cube,
        target_name: str,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "C N N N"]:
        rdt = 1.0 / dt
        return rdt * cube.old.cells[target_name].interior

    def source(
        self,
        cube: Cube,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "C N N N"]:
        rdt = 1.0 / dt
        return rdt * cube.old.cells[self._target_name].interior

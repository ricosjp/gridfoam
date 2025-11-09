import torch
from jaxtyping import Float

from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._field._grid import PyOctreeNode as Cube
from gridfoam._base._interface._fvm_term import IFVMTerm
from gridfoam._base._schemes._expr import FVMSchemeExpr
from gridfoam.utils.enums import DiscretizationMode


class LinearLaplacianScheme(IFVMTerm):
    """
    Laplacian term for finite volume method.

    This class represents the Laplacian term (∇·∇) in the finite
    volume method. It implements the spatial discretization for
    convective transport of a field variable.
    """

    def __init__(
        self,
        gamma_fd: FieldDescriptor,
        target_fd: FieldDescriptor,
        mode: DiscretizationMode,
    ):
        """
        Initialize the Laplacian term.

        Parameters
        ----------
        gamma_name : str
            Name of the diffusion coefficient field.
        fieldname : str
            Name of the field to compute Laplacian for.
        """
        self._target_name = target_fd.canonical_name
        self._gamma_name = gamma_fd.canonical_name
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
        if self._mode == DiscretizationMode.EXPLICIT:
            return torch.zeros_like(cube.old.cells[target_name].interior)

        dS = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        dV = dx[0] * dx[1] * dx[2]
        grad_f = cube.old.cells[target_name].grad(dx)
        nu_f = cube.old.cells[self._gamma_name].face_average()
        q_f = nu_f * grad_f
        q_f.x *= dS[0]
        q_f.y *= dS[1]
        q_f.z *= dS[2]
        return q_f.integrate_cell() / dV

    def source(
        self,
        cube: Cube,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "C N N N"]:
        if self._mode == DiscretizationMode.IMPLICIT:
            return torch.zeros_like(cube.old.cells[self._target_name].interior)

        dS = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        dV = dx[0] * dx[1] * dx[2]
        grad_f = cube.old.cells[self._target_name].grad(dx)
        nu_f = cube.old.cells[self._gamma_name].face_average()
        q_f = nu_f * grad_f
        q_f.x *= dS[0]
        q_f.y *= dS[1]
        q_f.z *= dS[2]
        return -q_f.integrate_cell() / dV

import torch
from jaxtyping import Float

from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._field._grid import PyOctreeNode as Cube
from gridfoam._base._field._tensor import FaceTensor
from gridfoam._base._interface._fvm_term import IFVMTerm
from gridfoam._base._schemes._div._tvd_schemes import Upwind
from gridfoam._base._schemes._expr import FVMSchemeExpr
from gridfoam.utils.enums import DiscretizationMode, Namespace


class UpwindDivScheme(IFVMTerm):
    """
    Divergence term for finite volume method.

    This class represents the divergence term (∇·) in the finite
    volume method. It implements the spatial discretization for
    convective transport of a field variable.
    """

    def __init__(
        self,
        velocity_fd: FieldDescriptor,
        target_fd: FieldDescriptor,
        mode: DiscretizationMode,
    ):
        """
        Initialize the divergence term.

        Parameters
        ----------
        velocity_fd : FieldDescriptor
            Field descriptor of the velocity field.
        target_fd : FieldDescriptor
            Field descriptor of the field to compute divergence for.
        mode : DiscretizationMode
            Mode of the discretization.
        """
        self._U_i_name = velocity_fd.canonical_name
        self._U_f_name = Namespace.RHIE_CHOW.value + "." + velocity_fd.name
        self._x_i_name = target_fd.canonical_name
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
        x_i = cube.old.cells[target_name]
        U_f = cube.old.faces[self._U_f_name]
        x_f = x_i.face_tensor_for_advection(U_f, scheme=Upwind())
        flow_rate_f = self._flow_rate(U_f, dS)
        q_f = x_f * flow_rate_f
        return q_f.integrate_cell() / dV

    def source(
        self,
        cube: Cube,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "C N N N"]:
        if self._mode == DiscretizationMode.IMPLICIT:
            return torch.zeros_like(cube.old.cells[self._x_i_name].interior)

        dS = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        dV = dx[0] * dx[1] * dx[2]
        x_i = cube.old.cells[self._x_i_name]
        U_f = cube.old.faces[self._U_f_name]
        x_f = x_i.face_tensor_for_advection(U_f, scheme=Upwind())
        flow_rate_f = self._flow_rate(U_f, dS)
        q_f = x_f * flow_rate_f
        return -q_f.integrate_cell() / dV

    def _flow_rate(
        self, U_f: FaceTensor, dS: Float[torch.Tensor, " 3"]
    ) -> FaceTensor:
        """
        Compute the flow rate tensor from velocity and face areas.

        Parameters
        ----------
        U_f : FaceTensor
            Face-centered velocity tensor.
        dS : Float[torch.Tensor, " 3"]
            Face areas in each direction.

        Returns
        -------
        FaceTensor
            Flow rate tensor (velocity * face area).
        """
        face_tensor = FaceTensor(U_f.N, 1, U_f.x.dtype, U_f.x.device)
        face_tensor.x[:] = U_f.x[0] * dS[0]
        face_tensor.y[:] = U_f.y[1] * dS[1]
        face_tensor.z[:] = U_f.z[2] * dS[2]
        return face_tensor

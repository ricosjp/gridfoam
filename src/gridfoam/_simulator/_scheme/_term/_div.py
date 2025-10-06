import torch
from jaxtyping import Float

from gridfoam._base import iter_leaf_cubes_of
from gridfoam._base._face_tensor import FaceTensor
from gridfoam._interface import IFVMTerm
from gridfoam._simulator._scheme._term._expr import Expr
from gridfoam.cubion import PyOctreeLevel
from gridfoam.utils.enums import TVDScheme


class Div(IFVMTerm):
    """
    Divergence term for finite volume method.

    This class represents the divergence term (∇·) in the finite
    volume method. It implements the spatial discretization for
    convective transport of a field variable.
    """

    def __init__(self, velocity_name: str, fieldname: str):
        """
        Initialize the divergence term.

        Parameters
        ----------
        velocity_name : str
            Name of the velocity field.
        fieldname : str
            Name of the field to compute divergence for.
        """
        self.fieldname = fieldname
        self.U_f_name = "_" + velocity_name + "_f"

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
        return Expr(self, other, "+")

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
        return Expr(self, other, "-")

    def matvec(
        self,
        octree_level: PyOctreeLevel,
        x: torch.Tensor,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Apply the divergence operator to a vector.

        Note: Currently returns zero as implicit formulation is not implemented.

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
            Result of applying the divergence operator (currently zero).
        """
        return torch.zeros_like(x)

    def diag(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Get the diagonal elements of the divergence operator.

        Note: Currently returns zero as implicit formulation is not implemented.

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
            Diagonal elements of the divergence operator (currently zero).
        """
        return torch.zeros(octree_level.n_leaf_cells)

    def rhs(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Compute the right-hand side vector for the divergence term.

        Note: Currently uses explicit formulation as implicit formulation
        across cubes and levels is not yet implemented.

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
            Right-hand side vector for the divergence term.
        """
        # NOTE: Implicit formulation across
        # cubes and levels is not yet implemented,
        # so this is currently an explicit formulation.
        dS = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        dV = dx[0] * dx[1] * dx[2]
        n_cells_per_cube = octree_level.n_cells_per_node
        b = torch.zeros(octree_level.n_leaf_cells)
        for i, cube in enumerate(iter_leaf_cubes_of(octree_level)):
            _slice = slice(i * n_cells_per_cube, (i + 1) * n_cells_per_cube)
            field_i = cube.old.cells[self.fieldname]
            U_f = cube.old.faces[self.U_f_name]
            field_f = field_i.face_tensor_for_advection(
                U_f, scheme=TVDScheme.VAN_LEER
            )
            flow_rate_f = self._flow_rate(U_f, dS)
            q_f = field_f * flow_rate_f
            b[_slice] = -q_f.integrate_cell().reshape(-1) / dV
        return b

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
            Flow rate tensor (velocity × face area).
        """
        return FaceTensor(
            U_f.w_interior, U_f.x[0] * dS[0], U_f.y[1] * dS[1], U_f.z[2] * dS[2]
        )

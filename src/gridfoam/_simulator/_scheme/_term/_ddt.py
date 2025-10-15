import torch
from jaxtyping import Float

from gridfoam._base import iter_leaf_cubes_of
from gridfoam._interface._fvmterm import IFVMTerm
from gridfoam._simulator._scheme._term._expr import Expr
from gridfoam.cubion import PyOctreeLevel


class Ddt(IFVMTerm):
    """
    Time derivative term for finite volume method.

    This class represents the time derivative term (∂/∂t) in the finite
    volume method. It implements the temporal discretization for a given
    field variable.
    """

    def __init__(self, fieldname: str):
        """
        Initialize the time derivative term.

        Parameters
        ----------
        fieldname : str
            Name of the field to be integrated over time.
        """
        self.fieldname = fieldname

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
        Apply the time derivative operator to a vector.

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
            Result of applying the time derivative operator.
        """
        depth = octree_level.depth
        rdt = (1 << depth) / dt
        return rdt * x

    def diag(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Get the diagonal elements of the time derivative operator.

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
            Diagonal elements of the time derivative operator.
        """
        depth = octree_level.depth
        rdt = (1 << depth) / dt
        return torch.full((octree_level.n_leaf_cells,), rdt)

    def rhs(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Compute the right-hand side vector for the time derivative term.

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
            Right-hand side vector for the time derivative term.
        """
        depth = octree_level.depth
        rdt = (1 << depth) / dt
        n_cells_per_cube = octree_level.n_cells_per_node
        b = torch.zeros(octree_level.n_leaf_cells)
        for i, cube in enumerate(iter_leaf_cubes_of(octree_level)):
            _slice = slice(i * n_cells_per_cube, (i + 1) * n_cells_per_cube)
            b[_slice] = rdt * cube.old.cells[self.fieldname].interior.reshape(
                -1
            )
        return b

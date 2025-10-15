import torch
from jaxtyping import Float

from gridfoam._base import iter_leaf_cubes_of
from gridfoam._base._cell_tensor import grad
from gridfoam._interface._fvmterm import IFVMTerm
from gridfoam._simulator._scheme._term._expr import Expr
from gridfoam.cubion import PyOctreeLevel


class Laplacian(IFVMTerm):
    """
    Laplacian term for finite volume method.

    This class represents the Laplacian term (∇·∇) in the finite
    volume method. It implements the spatial discretization for
    convective transport of a field variable.
    """

    def __init__(self, diffusion_name: str, fieldname: str):
        """
        Initialize the Laplacian term.

        Parameters
        ----------
        diffusion_name : str
            Name of the diffusion coefficient field.
        fieldname : str
            Name of the field to compute Laplacian for.
        """
        self.fieldname = fieldname
        self.diffusion_name = diffusion_name

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
        Apply the Laplacian operator to a vector.

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
            Result of applying the Laplacian operator (currently zero).
        """
        return torch.zeros_like(x)

    def diag(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Get the diagonal elements of the Laplacian operator.

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
            Diagonal elements of the Laplacian operator (currently zero).
        """
        return torch.zeros(octree_level.n_leaf_cells)

    def rhs(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Compute the right-hand side vector for the Laplacian term.

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
            Right-hand side vector for the Laplacian term.
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
            grad_field_f = grad(cube.old.cells[self.fieldname], dx)
            nu_f = cube.old.cells[self.diffusion_name].face_average()
            q_f = nu_f * grad_field_f
            q_f.x *= dS[0]
            q_f.y *= dS[1]
            q_f.z *= dS[2]
            b[_slice] = -q_f.integrate_cell().reshape(-1) / dV
        return b

import torch
from jaxtyping import Float

from gridfoam._interface import IFVMTerm
from gridfoam.cubion import PyOctreeLevel


class Expr(IFVMTerm):
    """
    Expression combining two FVM terms with an operator.

    This class represents a binary expression of two FVM terms
    combined with an operator (addition or subtraction).
    """

    def __init__(self, left: IFVMTerm, right: IFVMTerm, op: str):
        """
        Initialize the expression.

        Parameters
        ----------
        left : IFVMTerm
            Left operand of the expression.
        right : IFVMTerm
            Right operand of the expression.
        op : str
            Operator ("+" or "-").
        """
        self.left = left
        self.right = right
        self.op = op

    def matvec(
        self,
        octree_level: PyOctreeLevel,
        x: torch.Tensor,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Apply the expression operator to a vector.

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
            Result of applying the expression operator.

        Raises
        ------
        ValueError
            If the operator is not supported.
        """
        match self.op:
            case "+":
                return self.left.matvec(
                    octree_level, x, dt, dx
                ) + self.right.matvec(octree_level, x, dt, dx)
            case "-":
                return self.left.matvec(
                    octree_level, x, dt, dx
                ) - self.right.matvec(octree_level, x, dt, dx)
            case _:
                raise ValueError(f"Unknown operator: {self.op}")

    def diag(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Get the diagonal elements of the expression operator.

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
            Diagonal elements of the expression operator.
        """
        match self.op:
            case "+":
                return self.left.diag(octree_level, dt, dx) + self.right.diag(
                    octree_level, dt, dx
                )
            case "-":
                return self.left.diag(octree_level, dt, dx) - self.right.diag(
                    octree_level, dt, dx
                )
            case _:
                raise ValueError(f"Unknown operator: {self.op}")

    def rhs(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        """
        Compute the right-hand side vector for the expression.

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
            Right-hand side vector for the expression.

        Raises
        ------
        ValueError
            If the operator is not supported.
        """
        match self.op:
            case "+":
                return self.left.rhs(octree_level, dt, dx) + self.right.rhs(
                    octree_level, dt, dx
                )
            case "-":
                return self.left.rhs(octree_level, dt, dx) - self.right.rhs(
                    octree_level, dt, dx
                )
            case _:
                raise ValueError(f"Unknown operator: {self.op}")

import torch
from jaxtyping import Float

from gridfoam._base._field._grid import PyOctreeNode as Cube
from gridfoam._base._interface._fvm_term import IFVMTerm


class FVMSchemeExpr(IFVMTerm):
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
        """
        Apply the implicit term(A) to the input field(x).

        Parameters
        ----------
        cube : Cube
            The grid cube.
        target_name: str
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
        match self.op:
            case "+":
                return self.left.matvec(
                    cube, target_name, dt, dx
                ) + self.right.matvec(cube, target_name, dt, dx)
            case "-":
                return self.left.matvec(
                    cube, target_name, dt, dx
                ) - self.right.matvec(cube, target_name, dt, dx)
            case _:
                raise ValueError(f"Unknown operator: {self.op}")

    def source(
        self,
        cube: Cube,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
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
        match self.op:
            case "+":
                return self.left.source(cube, dt, dx) + self.right.source(
                    cube, dt, dx
                )
            case "-":
                return self.left.source(cube, dt, dx) - self.right.source(
                    cube, dt, dx
                )
            case _:
                raise ValueError(f"Unknown operator: {self.op}")

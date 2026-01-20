from gridfoam.DNA.ASTNodes._interface import IASTNode
from gridfoam.DNA.ASTNodes.operator_node import OperatorNode
from gridfoam.DNA.enum import OperatorType
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta


def ddt(psi: FieldMeta) -> IASTNode:
    return OperatorNode(type=OperatorType.DDT, args=[psi])


def div(a: FieldMeta, b: FieldMeta | None = None) -> IASTNode:
    """
    Compute the divergence of a field.
    If b is not provided, the divergence of a is computed.
    If b is provided, compute advection term considering a as the velocity field
    and b as the property which is transported.
        Example:
            div(U, T): calculate T transport by U.

    Parameters
    ----------
    a : FieldMeta
    b : FieldMeta | None = None
    """
    if b is None:
        return OperatorNode(type=OperatorType.DIV, args=[a])
    return OperatorNode(type=OperatorType.DIV, args=[a, b])


def laplacian(gamma: FieldMeta, psi: FieldMeta) -> IASTNode:
    return OperatorNode(type=OperatorType.LAPLACIAN, args=[gamma, psi])


def grad(psi: FieldMeta) -> IASTNode:
    return OperatorNode(type=OperatorType.GRAD, args=[psi])


# TODO: Implement trace tensor
# def tr(psi: FieldMeta) -> Expr:
#     return Expr(op="tr", args=[psi])

# TODO: Implement deviatoric and spherical tensors
# def dev(psi: FieldMeta) -> Expr:
#     return Expr(op="dev", args=[psi])

# def sph(psi: FieldMeta) -> Expr:
#     return Expr(op="sph", args=[psi])

# TODO: Implement symmetric and skew tensors
# def sym(psi: FieldMeta) -> Expr:
#     return Expr(op="sym", args=[psi])

# def skew(psi: FieldMeta) -> Expr:
#     return Expr(op="skew", args=[psi])


def equation(
    name: str,
    target: FieldMeta,
    boundary_conditions: list[BoundaryConditionMeta],
    lhs: IASTNode,
    rhs: IASTNode | None = None,
) -> EquationMeta:
    """
    Create an equation
    with a given name, lhs, rhs, target field, and boundary condition.

    Parameters
    ----------
    name : str
        Name of the equation.
    target : FieldMeta
        Target field of the equation.
        Note that the target field must be a state field.
    boundary_conditions : list[BoundaryConditionMeta]
        List of boundary conditions for the equation.
    lhs : ASTNode
        Left-hand side of the equation.
        Example:
            ddt(T) + div(U, T)
    rhs : ASTNode | None = None
        Right-hand side of the equation.
        If not provided, rhs is set to 0.

    Returns
    -------
    EquationMeta
        Equation meta.
    """
    return EquationMeta(
        name=name,
        target_field=target,
        boundary_conditions=boundary_conditions,
        ast_root=lhs - rhs,
    )

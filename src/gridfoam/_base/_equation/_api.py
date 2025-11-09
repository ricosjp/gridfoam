from gridfoam._base._equation._equation import Equation
from gridfoam._base._equation._expr import Expr
from gridfoam._base._field._descripter import FieldDescriptor


def ddt(psi: FieldDescriptor) -> Expr:
    return Expr(op_name="ddt", args=[psi])

def div(u: FieldDescriptor, psi: FieldDescriptor) -> Expr:
    return Expr(op_name="div", args=[u, psi])

def laplacian(gamma: FieldDescriptor, psi: FieldDescriptor) -> Expr:
    return Expr(op_name="laplacian", args=[gamma, psi])

# TODO: Implement trace tensor
# def tr(psi: FieldHandle) -> Expr:
#     return Expr(op="tr", args=[psi])

# TODO: Implement deviatoric and spherical tensors
# def dev(psi: FieldHandle) -> Expr:
#     return Expr(op="dev", args=[psi])

# def sph(psi: FieldHandle) -> Expr:
#     return Expr(op="sph", args=[psi])

# TODO: Implement symmetric and skew tensors
# def sym(psi: FieldHandle) -> Expr:
#     return Expr(op="sym", args=[psi])

# def skew(psi: FieldHandle) -> Expr:
#     return Expr(op="skew", args=[psi])


def eq_zero(arg: Expr, tag: str) -> Equation:
    return Equation(expr=arg, tag=tag)

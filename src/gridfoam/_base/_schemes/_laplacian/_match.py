from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._interface._fvm_term import IFVMTerm
from gridfoam._base._schemes._laplacian._linear import LinearLaplacianScheme
from gridfoam.config import SchemeChoice


def match_choice_to_laplacian_scheme(
    choice: SchemeChoice, gamma_fd: FieldDescriptor, target_fd: FieldDescriptor
) -> IFVMTerm:
    match choice.scheme:
        case "GaussLinear":
            return LinearLaplacianScheme(gamma_fd, target_fd, choice.mode)
        case _:
            raise ValueError(f"Unknown scheme: {choice.scheme}")

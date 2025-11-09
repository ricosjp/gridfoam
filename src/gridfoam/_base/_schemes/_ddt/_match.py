from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._interface._fvm_term import IFVMTerm
from gridfoam._base._schemes._ddt._euler import EulerDdtScheme
from gridfoam.config import SchemeChoice


def match_choice_to_ddt_scheme(
    choice: SchemeChoice, target_fd: FieldDescriptor
) -> IFVMTerm:
    match choice.scheme:
        case "Euler":
            return EulerDdtScheme(target_fd, choice.mode)
        case _:
            raise ValueError(f"Unknown scheme: {choice.scheme}")

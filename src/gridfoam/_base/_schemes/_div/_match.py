from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._interface._fvm_term import IFVMTerm
from gridfoam._base._schemes._div._upwind import UpwindDivScheme
from gridfoam.config import SchemeChoice


def match_choice_to_div_scheme(
    choice: SchemeChoice,
    velocity_fd: FieldDescriptor,
    target_fd: FieldDescriptor,
) -> IFVMTerm:
    match choice.scheme:
        case "upwind":
            return UpwindDivScheme(velocity_fd, target_fd, choice.mode)
        case _:
            raise ValueError(f"Unknown scheme: {choice.scheme}")

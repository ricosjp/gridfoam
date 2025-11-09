import abc

import torch
from jaxtyping import Float

from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._interface._fvm_term import IFVMTerm


class ISolver(abc.ABC):
    @abc.abstractmethod
    def solve(
        self,
        target_fd: FieldDescriptor,
        fvm_system: IFVMTerm,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> None:
        pass

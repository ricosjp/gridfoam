import abc

import torch
from jaxtyping import Float

from gridfoam.DNA.fielddata import CellField
from gridfoam.DNA.scheme.fvc._interface import IFVCOperator


class IFVCDivOperator(IFVCOperator):
    @classmethod
    @abc.abstractmethod
    def apply(
        cls,
        psi_c: CellField,
        dx: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "T 1 N N N"]:
        """
        Apply the FVC div operator.
        """
        pass

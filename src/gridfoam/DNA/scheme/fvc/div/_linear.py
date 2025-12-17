import torch
from jaxtyping import Float

from gridfoam.DNA.fielddata import CellField
from gridfoam.DNA.scheme.fvc.div._interface import IFVCDivOperator


class FVCDivLinear(IFVCDivOperator):
    @classmethod
    def apply(
        cls,
        psi_c: CellField,
        dx: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "T C N N N"]:
        Sf = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        V = dx[0] * dx[1] * dx[2]
        psi_f = psi_c.face_average()
        return psi_f.integrate_dSn(Sf) / V

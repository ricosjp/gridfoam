import torch
from jaxtyping import Float

from gridfoam.DNA.fielddata import CellField, FaceField
from gridfoam.DNA.scheme.fvc.div._interface import IFVCDivOperator


class FVCDivLinear(IFVCDivOperator):
    @classmethod
    def apply(
        cls,
        psi_c: CellField,
        dx: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "T 1 N N N"]:
        Sf = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        V = torch.prod(dx)
        # x: (T C N N L), y: (T C N L N), z: (T C L N N)
        psi_f = psi_c.face_average()
        psi_f_diag = FaceField(
            psi_c.T, 1, psi_c.N, psi_c.raw.dtype, psi_c.raw.device
        )
        psi_f_diag.x = psi_f.x[:, [0], :, :, :]  # (T 1 N N L)
        psi_f_diag.y = psi_f.y[:, [1], :, :, :]  # (T 1 N L N)
        psi_f_diag.z = psi_f.z[:, [2], :, :, :]  # (T 1 L N N)
        return psi_f_diag.integrate_dSn(Sf) / V  # (T 1 N N N)

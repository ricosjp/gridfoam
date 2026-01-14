import torch
from jaxtyping import Float

from gridfoam.DNA.fielddata import CellField
from gridfoam.DNA.scheme.fvc.grad._interface import IFVCGradOperator


class FVCGradLinear(IFVCGradOperator):
    @classmethod
    def apply(
        cls,
        psi_c: CellField,
        dx: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "T 3 C N N N"]:
        psi_f = psi_c.face_average()
        grad_x = (psi_f.x[:, :, :, :, 1:] - psi_f.x[:, :, :, :, :-1]) / dx[0]
        grad_y = (psi_f.y[:, :, :, 1:, :] - psi_f.y[:, :, :, :-1, :]) / dx[1]
        grad_z = (psi_f.z[:, :, 1:, :, :] - psi_f.z[:, :, :-1, :, :]) / dx[2]
        return torch.stack([grad_x, grad_y, grad_z], dim=1)

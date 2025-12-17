import torch
from jaxtyping import Float

from gridfoam.DNA.fielddata import CellField
from gridfoam.DNA.scheme.fvc.laplacian._interface import IFVCLaplacianOperator


class FVCLaplacianLinear(IFVCLaplacianOperator):
    @classmethod
    def apply(
        cls,
        gamma_c: CellField,
        psi_c: CellField,
        dx: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "T C N N N"]:
        # This method supports only scalar fields currently.
        assert gamma_c.C == 1
        assert psi_c.C == 1
        Sf = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        V = dx[0] * dx[1] * dx[2]
        gamma_f = gamma_c.face_average() # x: (T 1 N N L), y: (T 1 N L N), z: (T 1 L N N)
        grad_psi_f = psi_c.face_grad(dx) # x: (T 1 N N L), y: (T 1 N L N), z: (T 1 L N N)
        face_element = gamma_f * grad_psi_f
        return face_element.integrate_dSn(Sf) / V

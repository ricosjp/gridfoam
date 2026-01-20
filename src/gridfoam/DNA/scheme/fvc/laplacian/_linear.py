import torch
from jaxtyping import Float

from gridfoam.DNA.fielddata import CellField, FaceField
from gridfoam.DNA.scheme.fvc.laplacian._interface import IFVCLaplacianOperator


class FVCLaplacianLinear(IFVCLaplacianOperator):
    @classmethod
    def apply(
        cls,
        gamma_c: CellField,
        psi_c: CellField,
        dx: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "T 1 N N N"]:
        Sf = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        V = torch.prod(dx)
        # x: (T C N N L), y: (T C N L N), z: (T C L N N)
        gamma_f = gamma_c.face_harmonic_mean()
        grad_psi_f = psi_c.face_grad(dx)

        gamma_f_diag = FaceField(
            gamma_c.T, 1, gamma_c.N, gamma_c.raw.dtype, gamma_c.raw.device
        )
        if gamma_c.C == 1:
            gamma_f_diag.x = gamma_f.x
            gamma_f_diag.y = gamma_f.y
            gamma_f_diag.z = gamma_f.z
        elif gamma_c.C == 3:
            gamma_f_diag.x = gamma_f.x[:, [0], :, :, :]  # (T 1 N N L)
            gamma_f_diag.y = gamma_f.y[:, [1], :, :, :]  # (T 1 N L N)
            gamma_f_diag.z = gamma_f.z[:, [2], :, :, :]  # (T 1 L N N)
        else:
            raise ValueError(f"Invalid number of components: {gamma_c.C}")

        grad_psi_f_diag = FaceField(
            psi_c.T, 1, psi_c.N, psi_c.raw.dtype, psi_c.raw.device
        )
        if psi_c.C == 1:
            grad_psi_f_diag.x = grad_psi_f.x
            grad_psi_f_diag.y = grad_psi_f.y
            grad_psi_f_diag.z = grad_psi_f.z
        elif psi_c.C == 3:
            grad_psi_f_diag.x = grad_psi_f.x[:, [0], :, :, :]  # (T 1 N N L)
            grad_psi_f_diag.y = grad_psi_f.y[:, [1], :, :, :]  # (T 1 N L N)
            grad_psi_f_diag.z = grad_psi_f.z[:, [2], :, :, :]  # (T 1 L N N)
        else:
            raise ValueError(f"Invalid number of components: {psi_c.C}")

        face_element = gamma_f_diag * grad_psi_f_diag
        return face_element.integrate_dSn(Sf) / V  # (T 1 N N N)

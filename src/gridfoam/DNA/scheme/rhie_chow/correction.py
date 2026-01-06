import torch
from jaxtyping import Float

from gridfoam.DNA.fielddata import CellField, FaceField, FVMatrix


class RhieChowCorrection:
    @staticmethod
    def update_flux(
        phi_f: FaceField,
        p_c: CellField,
        fvmatrix: FVMatrix,
        dx: Float[torch.Tensor, " 3"],
    ) -> None:
        Sf = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        # (C N N L)
        phi_f.x[0] -= (Sf[0] / fvmatrix.a_fx[0:1]) * p_c.face_grad(dx).x[0]
        # (C N L N)
        phi_f.y[0] -= (Sf[1] / fvmatrix.a_fy[1:2]) * p_c.face_grad(dx).y[0]
        # (C L N N)
        phi_f.z[0] -= (Sf[2] / fvmatrix.a_fz[2:3]) * p_c.face_grad(dx).z[0]

    @staticmethod
    def update_velocity(
        U_c: CellField,
        p_c: CellField,
        fvmatrix: FVMatrix,
        dx: Float[torch.Tensor, " 3"],
    ) -> None:
        p_f = p_c.face_average()
        gradp_x = (p_f.x[0, 0, :, :, 1:] - p_f.x[0, 0, :, :, :-1]) / dx[0]
        gradp_y = (p_f.y[0, 0, :, 1:, :] - p_f.y[0, 0, :, :-1, :]) / dx[1]
        gradp_z = (p_f.z[0, 0, 1:, :, :] - p_f.z[0, 0, :-1, :, :]) / dx[2]
        gradp = torch.stack([gradp_x, gradp_y, gradp_z], dim=0) # (3 N N N)
        U_c.interior[0] -= (1.0 / fvmatrix.a_P.interior[0]) * gradp # (3 N N N)

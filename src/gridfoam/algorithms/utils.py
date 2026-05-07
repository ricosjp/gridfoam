import torch

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid


def needs_reference_value(field: CellField) -> bool:
    """
    Check if the field needs a reference value to solve the equation.
    """
    for bc in field.bcs.values():
        if isinstance(bc, DirichletBC):
            return False
    return True


def set_reference_value(mat: FvMatrix):
    """
    Apply a pressure reference in OpenFOAM setReference style.
    """
    ref_cell = mat.field.ref_cell_id
    ref_diag = mat.diag[ref_cell].clone()
    mat.diag[ref_cell] += ref_diag
    mat.source[ref_cell] += ref_diag * mat.field.ref_value




def apply_pressure_flux_correction(
    phi: FaceField, rAU_f: FaceField, sn_grad_p: FaceField
) -> None:
    """Apply pressure-gradient correction to face flux."""
    grid = phi.grid
    mag_Sf_single = torch.linalg.vector_norm(
        grid.Sf[phi.single_mask], dim=1, keepdim=True
    )
    mag_Sf_domain = torch.linalg.vector_norm(
        grid.domain_bnd_Sf, dim=1, keepdim=True
    )
    phi.single_data = (
        phi.single_data
        - rAU_f.single_data * mag_Sf_single * sn_grad_p.single_data
    )
    phi.domain_bnd_data = (
        phi.domain_bnd_data
        - rAU_f.domain_bnd_data * mag_Sf_domain * sn_grad_p.domain_bnd_data
    )
    if isinstance(grid, AxisProjectedGrid):
        mag_Sf_immersed = torch.linalg.vector_norm(
            grid.Sf[grid.ap_is_immersed_faces], dim=1, keepdim=True
        )
        phi.immersed_upper = (
            phi.immersed_upper
            - rAU_f.immersed_upper * mag_Sf_immersed * sn_grad_p.immersed_upper
        )
        phi.immersed_lower = (
            phi.immersed_lower
            - rAU_f.immersed_lower * mag_Sf_immersed * sn_grad_p.immersed_lower
        )


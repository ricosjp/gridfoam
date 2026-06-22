from collections.abc import Callable

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField, FaceField
from gridfoam.fv import fvc
from gridfoam.meta.enums import DivScheme

DivSchemeFunc = Callable[
    [FaceField, CellField],
    tuple[
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single k"],
    ],
]
LimiterFunc = Callable[
    [Float[torch.Tensor, " F_single 1"]],
    Float[torch.Tensor, " F_single 1"],
]


def upwind(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single k"],
]:
    """
    First-order upwind differencing scheme.

    Parameters
    ----------
    phi : FaceField
        Face mass/volumetric flux field.
    field : CellField
        Target field.

    Returns
    -------
    tuple[
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single k"],
    ]
        (upper, lower, diag_owner, diag_neighbour, source_face)
    """
    pos_flux = torch.clamp(phi.single_data, min=0.0)
    neg_flux = torch.clamp(phi.single_data, max=0.0)

    upper = neg_flux
    lower = -pos_flux
    diag_O = pos_flux
    diag_N = -neg_flux

    # Explicit source contribution (zero deferred correction for upwind).
    shape = (phi.num_single_sided, field.num_components)
    source_face = torch.zeros(
        shape, dtype=phi.grid.dtype, device=phi.grid.device
    )  # [F_single k]

    return upper, lower, diag_O, diag_N, source_face


def linear(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single k"],
]:
    """
    Second-order linear (central differencing) scheme.

    Parameters
    ----------
    phi : FaceField
        Face flux driving convection.
    field : CellField
        Target field.

    Returns
    -------
    tuple[
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single k"],
    ]
        (upper, lower, diag_owner, diag_neighbour, source_face)
    """
    grid = field.grid
    owner = grid.owner[phi.single_mask]
    neighbour = grid.neighbour[phi.single_mask]

    # Compute linear interpolation weight w at each face:
    # phi_f = w * phi_O + (1 - w) * phi_N.
    axis_idx = grid.axis[phi.single_mask, None]
    d_ON_vec = grid.cell_centers[neighbour] - grid.cell_centers[owner]
    d_fN_vec = grid.cell_centers[neighbour] - grid.face_centers[phi.single_mask]

    d_ON = torch.abs(d_ON_vec.gather(1, axis_idx))
    d_fN = torch.abs(d_fN_vec.gather(1, axis_idx))
    w = d_fN / d_ON

    # Flux contribution to owner equation:
    # +flux * (w * phi_O + (1 - w) * phi_N)
    diag_O = phi.single_data * w
    upper = phi.single_data * (1.0 - w)

    # Flux contribution to neighbor equation:
    # -flux * (w * phi_O + (1 - w) * phi_N)
    lower = -phi.single_data * w
    diag_N = -phi.single_data * (1.0 - w)

    # Pure central differencing has zero deferred-correction source.
    shape = (phi.num_single_sided, field.num_components)
    source_face = torch.zeros(
        shape, dtype=phi.grid.dtype, device=phi.grid.device
    )  # [F_single k]

    return upper, lower, diag_O, diag_N, source_face


# =============================================================================
# TVD (Total Variation Diminishing) schemes
# =============================================================================
def limiter_vanleer(
    r: Float[torch.Tensor, " F_single 1"],
) -> Float[torch.Tensor, " F_single 1"]:
    """Van Leer limiter."""
    return (r + torch.abs(r)) / (1.0 + torch.abs(r))


def limiter_minmod(
    r: Float[torch.Tensor, " F_single 1"],
) -> Float[torch.Tensor, " F_single 1"]:
    """Minmod limiter."""
    return torch.clamp(torch.minimum(torch.ones_like(r), r), min=0.0)


def limiter_superbee(
    r: Float[torch.Tensor, " F_single 1"],
) -> Float[torch.Tensor, " F_single 1"]:
    """SuperBee limiter."""
    r2 = 2.0 * r
    v1 = torch.minimum(r2, torch.ones_like(r))
    v2 = torch.minimum(r, torch.full_like(r, 2.0))
    return torch.clamp(torch.maximum(v1, v2), min=0.0)


def limiter_monotonized_central(
    r: Float[torch.Tensor, " F_single 1"],
) -> Float[torch.Tensor, " F_single 1"]:
    """Monotonized Central limiter."""
    v1 = 2.0 * r
    v2 = 0.5 * r + 0.5
    v3 = torch.full_like(r, 2.0)
    return torch.clamp(torch.minimum(torch.minimum(v1, v2), v3), min=0.0)


def limiter_limitedlinear(
    r: Float[torch.Tensor, " F_single 1"],
) -> Float[torch.Tensor, " F_single 1"]:
    """
    limitedLinear limiter (equivalent to OpenFOAM limitedLinear 1).
    """
    return torch.clamp(torch.minimum(2.0 * r, torch.ones_like(r)), min=0.0)


def limiter_koren(
    r: Float[torch.Tensor, " F_single 1"],
) -> Float[torch.Tensor, " F_single 1"]:
    """Koren limiter."""
    v1 = 2.0 * r
    v2 = (2.0 + r) / 3.0
    v3 = torch.full_like(r, 2.0)
    return torch.clamp(torch.minimum(torch.minimum(v1, v2), v3), min=0.0)


def _r_for_vector_field(
    gradf_v: Float[torch.Tensor, " F_single k"],
    d: Float[torch.Tensor, " F_single 3"],
    grad_tensor_upwind: Float[torch.Tensor, " F_single k 3"],
) -> Float[torch.Tensor, " F_single 1"]:
    """
    Normalized TVD variable corresponding to OpenFOAM ``NVDVTVDV::r``.

    Parameters
    ----------
    gradf_v : Float[torch.Tensor, " F_single k"]
        Face value difference ``psiN - psiP``.
    d : Float[torch.Tensor, " F_single 3"]
        Cell-center displacement vector.
    grad_tensor_upwind : Float[torch.Tensor, " F_single k 3"]
        Upwind-cell gradient tensor ``dU_i/dx_j``.

    Returns
    -------
    Float[torch.Tensor, " F_single 1"]
        Scalar ``r`` for each face.
    """
    gradf = torch.sum(gradf_v * gradf_v, dim=1, keepdim=True)  # [F_single 1]
    d_grad_t_u = torch.sum(
        d[:, None, :] * grad_tensor_upwind, dim=2
    )  #  [F_single k]
    gradcf = torch.sum(
        gradf_v * d_grad_t_u, dim=1, keepdim=True
    )  # [F_single 1]

    mag_gradf = torch.abs(gradf)
    mag_gradcf = torch.abs(gradcf)
    # Limit the gradient ratio to avoid overflow when gradf is too small
    # https://cpp.openfoam.org/v10/NVDVTVDV_8H_source.html
    steep = mag_gradcf >= 1000.0 * mag_gradf
    return torch.where(
        steep,
        2.0 * 1000.0 * torch.sign(gradcf) * torch.sign(gradf) - 1.0,
        2.0 * gradcf / gradf - 1.0,
    )


def _apply_tvd_scheme(
    phi: FaceField,
    field: CellField,
    limiter_func: LimiterFunc,
) -> tuple[
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single k"],
]:
    """
    Common deferred-correction routine for TVD schemes.

    For vector fields, this follows the same formulation style as
    OpenFOAM ``*V`` variants such as ``limitedLinearV``.

    Parameters
    ----------
    phi : FaceField
        Face mass/volumetric flux field.
    field : CellField
        Target field.
    limiter_func : Callable
        Limiter function mapping gradient ratio ``r`` to limiter value.

    Returns
    -------
    tuple[
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single 1"],
        Float[torch.Tensor, " F_single k"],
    ]
        (upper, lower, diag_owner, diag_neighbour, source_face)
    """
    grid = field.grid
    owner = grid.owner[phi.single_mask]
    neighbour = grid.neighbour[phi.single_mask]

    # 1. Build stable upwind matrix coefficients.
    pos_flux = torch.clamp(phi.single_data, min=0.0)
    neg_flux = torch.clamp(phi.single_data, max=0.0)

    upper = neg_flux  # [F_single 1]
    lower = -pos_flux  # [F_single 1]
    diag_O = pos_flux  # [F_single 1]
    diag_N = -neg_flux  # [F_single 1]

    # 2. Gather linear, upwind, and downwind face states.
    axis_idx = grid.axis[phi.single_mask, None]
    d_ON_vec = grid.cell_centers[neighbour] - grid.cell_centers[owner]
    d_fN_vec = grid.cell_centers[neighbour] - grid.face_centers[phi.single_mask]

    d_ON_mag = torch.abs(d_ON_vec.gather(1, axis_idx))  # [F_single 1]
    d_fN_mag = torch.abs(d_fN_vec.gather(1, axis_idx))  # [F_single 1]
    w = d_fN_mag / d_ON_mag

    psi_O = field.data[owner]  # [F_single k]
    psi_N = field.data[neighbour]  # [F_single k]

    psi_linear = w * psi_O + (1.0 - w) * psi_N  # [F_single k]
    psi_upwind = torch.where(phi.single_data > 0, psi_O, psi_N)  # [F_single k]

    # 3. Compute gradients and OpenFOAM-style NVDTVD/NVDVTVDV r.
    n_cells = grid.num_cells

    if field.num_components == 1:
        grad_data = fvc.grad(field).data  # [n_cells, 3]
        grad_O = grad_data[owner]  # [F_single, 3]
        grad_N = grad_data[neighbour]  # [F_single, 3]
        flux_mask = phi.single_data[:, 0] > 0  # [F_single]
        # Select upwind gradient.
        grad_U = torch.where(
            flux_mask[:, None], grad_O, grad_N
        )  # [F_single, 3]
        grad_U_dot_d = torch.sum(
            grad_U * d_ON_vec, dim=1, keepdim=True
        )  # [F_single, 1]

        gradf = psi_N - psi_O  # [F_single k] (k=1)
        gradcf = grad_U_dot_d

        mag_gradf = torch.abs(gradf)
        mag_gradcf = torch.abs(gradcf)
        steep = mag_gradcf >= 1000.0 * mag_gradf
        r = torch.where(
            steep,
            2.0 * 1000.0 * torch.sign(gradcf) * torch.sign(gradf) - 1.0,
            2.0 * gradcf / gradf - 1.0,
        )  # [F_single, 1]

    else:
        grad_data = fvc.grad(field).data.reshape(
            n_cells, field.num_components, 3
        )
        grad_t_O = grad_data[owner]  # [F_single k 3]
        grad_t_N = grad_data[neighbour]  # [F_single k 3]
        flux_mask = phi.single_data > 0  # [F_single 1]
        # Select upwind gradient tensor.
        grad_t_u = torch.where(
            flux_mask[:, None], grad_t_O, grad_t_N
        )  # [F_single k 3]
        # Face value difference.
        gradf_v = psi_N - psi_O
        r = _r_for_vector_field(gradf_v, d_ON_vec, grad_t_u)  # [F_single 1]

    # 4. Apply limiter.
    limiter = limiter_func(r)  # [F_single 1]

    # 5. Compute deferred-correction source:
    # added flux = flux * limiter * (psi_linear - psi_upwind)
    # moved from LHS to RHS with negative sign.
    source_face = (
        -phi.single_data * limiter * (psi_linear - psi_upwind)
    )  # [F_single k]

    return upper, lower, diag_O, diag_N, source_face


def vanleer(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single k"],
]:
    """
    Van Leer TVD scheme.
    """
    return _apply_tvd_scheme(phi, field, limiter_vanleer)


def minmod(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single k"],
]:
    """
    Minmod TVD scheme.
    """
    return _apply_tvd_scheme(phi, field, limiter_minmod)


def superbee(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single k"],
]:
    """
    SuperBee TVD scheme.
    """
    return _apply_tvd_scheme(phi, field, limiter_superbee)


def monotonized_central(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single k"],
]:
    """
    Monotonized Central TVD scheme.
    """
    return _apply_tvd_scheme(phi, field, limiter_monotonized_central)


def limitedlinear(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single k"],
]:
    """
    limitedLinear TVD scheme.

    Equivalent to OpenFOAM limitedLinear 1.
    """
    return _apply_tvd_scheme(phi, field, limiter_limitedlinear)


def koren(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single 1"],
    Float[torch.Tensor, " F_single k"],
]:
    """
    Koren TVD scheme.
    """
    return _apply_tvd_scheme(phi, field, limiter_koren)


# Scheme dispatch table
DIV_SCHEMES: dict[DivScheme, DivSchemeFunc] = {
    DivScheme.UPWIND: upwind,
    DivScheme.LINEAR: linear,
    DivScheme.VANLEER: vanleer,
    DivScheme.MINMOD: minmod,
    DivScheme.SUPERBEE: superbee,
    DivScheme.MONOTONIZED_CENTRAL: monotonized_central,
    DivScheme.LIMITEDLINEAR: limitedlinear,
    DivScheme.KOREN: koren,
}


def get_div_scheme(scheme: DivScheme) -> DivSchemeFunc:
    """Return the divergence-scheme function for the given enum."""
    return DIV_SCHEMES[scheme]

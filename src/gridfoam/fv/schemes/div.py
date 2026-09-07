from collections.abc import Callable

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.shapes import (
    broadcast_entity,
    sum_physical,
)
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.fv.kernels.face_interpolation import (
    correct_internal_values,
    linear_internal_face_values,
)
from gridfoam.fv.kernels.least_squares import least_squares_gradient
from gridfoam.fv.schemes.grad import eval_grad
from gridfoam.meta.enums import DivScheme

DivSchemeFunc = Callable[
    [FaceField, CellField],
    tuple[
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single *component_shape"],
    ],
]
LimiterFunc = Callable[
    [Float[torch.Tensor, " F_single"]],
    Float[torch.Tensor, " F_single"],
]


def upwind(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single *component_shape"],
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
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single *component_shape"],
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
    shape = (phi.num_single_sided, *field.component_shape)
    source_face = torch.zeros(
        shape, dtype=phi.grid.dtype, device=phi.grid.device
    )  # [F_single *component_shape]

    return upper, lower, diag_O, diag_N, source_face


def linear(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single *component_shape"],
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
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single *component_shape"],
    ]
        (upper, lower, diag_owner, diag_neighbour, source_face)
    """
    geo = face_geometry(field.grid)
    w = geo.w_s

    # Flux contribution to owner equation:
    # +flux * (w * phi_O + (1 - w) * phi_N)
    diag_O = phi.single_data * w
    upper = phi.single_data * (1.0 - w)

    # Flux contribution to neighbor equation:
    # -flux * (w * phi_O + (1 - w) * phi_N)
    lower = -phi.single_data * w
    diag_N = -phi.single_data * (1.0 - w)

    # Face-centre offset correction on hanging faces as an explicit
    # deferred source; zero on regular faces.
    shape = (phi.num_single_sided, *field.component_shape)
    source_face = torch.zeros(
        shape, dtype=phi.grid.dtype, device=phi.grid.device
    )
    if geo.num_hanging > 0:
        base_values = linear_internal_face_values(field, geo)
        grad_hang = least_squares_gradient(field, geo, hanging_cells_only=True)
        corrected = correct_internal_values(field, base_values, grad_hang, geo)
        flux = broadcast_entity(phi.single_data, base_values)
        source_face = -flux * (corrected - base_values)

    return upper, lower, diag_O, diag_N, source_face


# =============================================================================
# TVD (Total Variation Diminishing) schemes
# =============================================================================
def limiter_vanleer(
    r: Float[torch.Tensor, " F_single"],
) -> Float[torch.Tensor, " F_single"]:
    """Van Leer limiter."""
    return (r + torch.abs(r)) / (1.0 + torch.abs(r))


def limiter_minmod(
    r: Float[torch.Tensor, " F_single"],
) -> Float[torch.Tensor, " F_single"]:
    """Minmod limiter."""
    return torch.clamp(torch.minimum(torch.ones_like(r), r), min=0.0)


def limiter_superbee(
    r: Float[torch.Tensor, " F_single"],
) -> Float[torch.Tensor, " F_single"]:
    """SuperBee limiter."""
    r2 = 2.0 * r
    v1 = torch.minimum(r2, torch.ones_like(r))
    v2 = torch.minimum(r, torch.full_like(r, 2.0))
    return torch.clamp(torch.maximum(v1, v2), min=0.0)


def limiter_monotonized_central(
    r: Float[torch.Tensor, " F_single"],
) -> Float[torch.Tensor, " F_single"]:
    """Monotonized Central limiter."""
    v1 = 2.0 * r
    v2 = 0.5 * r + 0.5
    v3 = torch.full_like(r, 2.0)
    return torch.clamp(torch.minimum(torch.minimum(v1, v2), v3), min=0.0)


def limiter_limitedlinear(
    r: Float[torch.Tensor, " F_single"],
) -> Float[torch.Tensor, " F_single"]:
    """
    limitedLinear limiter (equivalent to OpenFOAM limitedLinear 1).
    """
    return torch.clamp(torch.minimum(2.0 * r, torch.ones_like(r)), min=0.0)


def limiter_koren(
    r: Float[torch.Tensor, " F_single"],
) -> Float[torch.Tensor, " F_single"]:
    """Koren limiter."""
    v1 = 2.0 * r
    v2 = (2.0 + r) / 3.0
    v3 = torch.full_like(r, 2.0)
    return torch.clamp(torch.minimum(torch.minimum(v1, v2), v3), min=0.0)


def _r_for_multicomponent_field(
    gradf_v: Float[torch.Tensor, " F_single *component_shape"],
    d: Float[torch.Tensor, " F_single 3"],
    grad_tensor_upwind: Float[torch.Tensor, " F_single *component_shape 3"],
) -> Float[torch.Tensor, " F_single"]:
    """
    Normalized TVD variable corresponding to OpenFOAM ``NVDVTVDV::r``.

    Parameters
    ----------
    gradf_v : Float[torch.Tensor, " F_single *component_shape"]
        Face value difference ``psiN - psiP``.
    d : Float[torch.Tensor, " F_single 3"]
        Cell-center displacement vector.
    grad_tensor_upwind : Float[torch.Tensor, " F_single *component_shape 3"]
        Upwind-cell gradient tensor ``dU_i/dx_j``.

    Returns
    -------
    Float[torch.Tensor, " F_single"]
        Scalar ``r`` for each face.
    """
    gradf = sum_physical(gradf_v * gradf_v)  # [F_single]
    # Contract only the differentiation direction, retaining field components.
    d_grad_t_u = torch.einsum("n...j,nj->n...", grad_tensor_upwind, d)
    gradcf = sum_physical(gradf_v * d_grad_t_u)  # [F_single]

    mag_gradf = torch.abs(gradf)
    mag_gradcf = torch.abs(gradcf)
    # Limit the gradient ratio to avoid overflow when gradf is too small
    # https://cpp.openfoam.org/v10/NVDVTVDV_8H_source.html
    steep = mag_gradcf >= 1000.0 * mag_gradf
    # Mask before division: torch.where evaluates both branches, and an
    # unused 0/0 would still produce NaN gradients during backward.
    safe_gradf = torch.where(steep, torch.ones_like(gradf), gradf)
    return torch.where(
        steep,
        2.0 * 1000.0 * torch.sign(gradcf) * torch.sign(gradf) - 1.0,
        2.0 * gradcf / safe_gradf - 1.0,
    )


def _apply_tvd_scheme(
    phi: FaceField,
    field: CellField,
    limiter_func: LimiterFunc,
) -> tuple[
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single *component_shape"],
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
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single"],
        Float[torch.Tensor, " F_single *component_shape"],
    ]
        (upper, lower, diag_owner, diag_neighbour, source_face)
    """
    grid = field.grid
    geo = face_geometry(grid)
    owner, neighbour, w = geo.owner_s, geo.neighbour_s, geo.w_s
    d_ON_vec = grid.cell_centers[neighbour] - grid.cell_centers[owner]

    # 1. Build stable upwind matrix coefficients.
    pos_flux = torch.clamp(phi.single_data, min=0.0)
    neg_flux = torch.clamp(phi.single_data, max=0.0)

    upper = neg_flux  # [F_single]
    lower = -pos_flux  # [F_single]
    diag_O = pos_flux  # [F_single]
    diag_N = -neg_flux  # [F_single]

    # 2. Gather linear, upwind, and downwind face states.
    psi_O = field.data[owner]  # [F_single *component_shape]
    psi_N = field.data[neighbour]  # [F_single *component_shape]

    flux_mask = phi.single_data > 0
    value_mask = broadcast_entity(flux_mask, psi_O)
    psi_upwind = torch.where(value_mask, psi_O, psi_N)

    # 3. Compute gradients and OpenFOAM-style NVDTVD/NVDVTVDV r.
    # The configured gradient is needed on every upwind cell for ``r``; it
    # is reused for the hanging-face offset correction of the linear value.
    grad_tensor = eval_grad(field)
    w_view = broadcast_entity(w, psi_O)
    base_values = w_view * psi_O + (1.0 - w_view) * psi_N
    psi_linear = correct_internal_values(field, base_values, grad_tensor, geo)

    if field.component_shape == ():
        grad_O = grad_tensor[owner]  # [F_single, 3]
        grad_N = grad_tensor[neighbour]  # [F_single, 3]
        # Select upwind gradient.
        grad_U = torch.where(
            flux_mask[:, None], grad_O, grad_N
        )  # [F_single, 3]
        grad_U_dot_d = torch.sum(grad_U * d_ON_vec, dim=1)  # [F_single]

        gradf = psi_N - psi_O  # [F_single]
        gradcf = grad_U_dot_d

        mag_gradf = torch.abs(gradf)
        mag_gradcf = torch.abs(gradcf)
        steep = mag_gradcf >= 1000.0 * mag_gradf
        # Avoid zero division in the unselected branch during backward.
        safe_gradf = torch.where(steep, torch.ones_like(gradf), gradf)
        r = torch.where(
            steep,
            2.0 * 1000.0 * torch.sign(gradcf) * torch.sign(gradf) - 1.0,
            2.0 * gradcf / safe_gradf - 1.0,
        )  # [F_single]

    else:
        grad_t_O = grad_tensor[owner]  # [F_single *component_shape 3]
        grad_t_N = grad_tensor[neighbour]  # [F_single *component_shape 3]
        # Select upwind gradient tensor.
        gradient_mask = broadcast_entity(flux_mask, grad_t_O)
        grad_t_u = torch.where(gradient_mask, grad_t_O, grad_t_N)
        # Face value difference.
        gradf_v = psi_N - psi_O
        r = _r_for_multicomponent_field(
            gradf_v, d_ON_vec, grad_t_u
        )  # [F_single]

    # 4. Apply limiter.
    limiter = limiter_func(r)  # [F_single]

    # 5. Compute deferred-correction source:
    # added flux = flux * limiter * (psi_linear - psi_upwind)
    # moved from LHS to RHS with negative sign.
    limited_flux = broadcast_entity(phi.single_data * limiter, psi_linear)
    source_face = -limited_flux * (psi_linear - psi_upwind)

    return upper, lower, diag_O, diag_N, source_face


def vanleer(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single *component_shape"],
]:
    """
    Van Leer TVD scheme.
    """
    return _apply_tvd_scheme(phi, field, limiter_vanleer)


def minmod(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single *component_shape"],
]:
    """
    Minmod TVD scheme.
    """
    return _apply_tvd_scheme(phi, field, limiter_minmod)


def superbee(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single *component_shape"],
]:
    """
    SuperBee TVD scheme.
    """
    return _apply_tvd_scheme(phi, field, limiter_superbee)


def monotonized_central(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single *component_shape"],
]:
    """
    Monotonized Central TVD scheme.
    """
    return _apply_tvd_scheme(phi, field, limiter_monotonized_central)


def limitedlinear(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single *component_shape"],
]:
    """
    limitedLinear TVD scheme.

    Equivalent to OpenFOAM limitedLinear 1.
    """
    return _apply_tvd_scheme(phi, field, limiter_limitedlinear)


def koren(
    phi: FaceField, field: CellField
) -> tuple[
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single *component_shape"],
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

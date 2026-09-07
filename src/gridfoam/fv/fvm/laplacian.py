import math
from typing import overload

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.shapes import broadcast_entity, require_shape
from gridfoam.fv.boundary_ops import (
    boundary_value_gradient_coefficient,
    evaluate_boundary_state,
    iter_boundary_batches,
    outward_boundary_Sf,
)
from gridfoam.fv.kernels.face_geometry import FaceGeometry, face_geometry
from gridfoam.fv.kernels.face_interpolation import sn_grad_hanging_correction
from gridfoam.fv.kernels.least_squares import least_squares_gradient
from gridfoam.meta.config import SimulatorConfig
from gridfoam.meta.enums import LaplacianScheme

DEFAULT_LAPLACIAN_SCHEME = LaplacianScheme.CORRECTED


def search_laplacian_scheme(
    sim_config: SimulatorConfig, field: CellField
) -> LaplacianScheme:
    """
    Resolve the ``laplacianSchemes`` entry for ``field``.

    Lookup order is ``laplacian(<field>)``, then ``default``, then
    :data:`DEFAULT_LAPLACIAN_SCHEME`. ``LINEAR`` is treated as
    ``CORRECTED``.
    """
    schemes = sim_config.fvSchemes.laplacianSchemes
    if schemes is None:
        return DEFAULT_LAPLACIAN_SCHEME
    scheme = schemes.get(f"laplacian({field.name})")
    if scheme is None:
        scheme = schemes.get("default")
    if scheme is None:
        return DEFAULT_LAPLACIAN_SCHEME
    if scheme in (
        LaplacianScheme.LINEAR,
        LaplacianScheme.GAUSS_LINEAR_CORRECTED,
    ):
        return LaplacianScheme.CORRECTED
    if scheme == LaplacianScheme.GAUSS_LINEAR_UNCORRECTED:
        return LaplacianScheme.UNCORRECTED
    return scheme


@overload
def _interpolate_gamma(
    geo: FaceGeometry,
    gamma: Float[torch.Tensor, " C"],
    *,
    harmonic: bool = False,
) -> Float[torch.Tensor, " F"]: ...


@overload
def _interpolate_gamma(
    geo: FaceGeometry, gamma: float, *, harmonic: bool = False
) -> float: ...


def _interpolate_gamma(
    geo: FaceGeometry,
    gamma: Float[torch.Tensor, " C"] | float,
    *,
    harmonic: bool = False,
) -> Float[torch.Tensor, " F"] | float:
    """Interpolate scalar diffusivity, optionally as series resistances."""
    if harmonic:
        valid = (
            bool(torch.all(torch.isfinite(gamma) & (gamma >= 0)))
            if isinstance(gamma, torch.Tensor)
            else math.isfinite(gamma) and gamma >= 0
        )
        if not valid:
            raise ValueError(
                "Harmonic diffusivity must be finite and nonnegative"
            )
    if not isinstance(gamma, torch.Tensor):
        return gamma
    # All internal faces, including immersed; those coefficients are zeroed
    # later so the LDU arrays keep internal-face length.
    gamma_o, gamma_n = gamma[geo.owner], gamma[geo.neighbour]
    w = geo.w_all
    if not harmonic:
        return w * gamma_o + (1.0 - w) * gamma_n

    # w = d_Nf / (d_Of + d_Nf), so the resistance weights are reversed:
    # gamma_f = 1 / ((1-w)/gamma_O + w/gamma_N).
    # Scale before forming products or reciprocals to avoid overflow.
    # Zero diffusivity blocks the face; mask denominators before division
    # so the all-zero case also has finite autograd derivatives.
    scale = torch.maximum(gamma_o, gamma_n)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    denom = w * (gamma_o / safe_scale) + (1.0 - w) * (gamma_n / safe_scale)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return torch.minimum(gamma_o, gamma_n) / safe_denom


def _hanging_correction_source(
    field: CellField,
    geo: FaceGeometry,
    gamma_f: Float[torch.Tensor, " F"] | float,
) -> Float[torch.Tensor, " F_hang *component_shape"]:
    """
    Explicit skewness-correction flux on hanging-node faces.

    Returns ``gamma_f * |Sf| * (snGrad_corrected - snGrad_uncorrected)``
    using the same correction kernel as ``fvc.sn_grad``, so the matrix
    flux and the explicit surface-normal gradient agree exactly.
    """
    grad_hang = least_squares_gradient(field, geo, hanging_cells_only=True)
    correction = sn_grad_hanging_correction(field, grad_hang, geo)
    hang_faces = geo.single_idx[geo.hang_idx]
    if isinstance(gamma_f, torch.Tensor):
        gamma_hang = gamma_f[hang_faces]
    else:
        gamma_hang = gamma_f
    coeff = broadcast_entity(
        gamma_hang * geo.mag_Sf_s[geo.hang_idx], correction
    )
    return coeff * correction


def laplacian(
    gamma: Float[torch.Tensor, " C"] | float, field: CellField
) -> FvMatrix:
    """
    Build the diffusion (Laplacian) matrix term.

    Represents div(gamma * grad(phi)). The implicit part uses the
    orthogonal two-point stencil ``gamma_f |Sf| / |d . n|`` on every face,
    which keeps the matrix symmetric. On an octree the only
    non-orthogonal/skewed faces are hanging-node (2:1) interfaces; with the
    ``corrected`` scheme their skewness correction is added as an explicit
    source and recorded in ``FvMatrix.face_flux_correction``. The
    ``uncorrected`` scheme omits that term.

    Coefficients use linear interpolation by default. The configured
    ``Gauss harmonic corrected`` / ``Gauss harmonic uncorrected`` options
    use the series-resistance mean for face-aligned material interfaces.
    Harmonic coefficients must be finite and nonnegative; a zero cell
    coefficient blocks diffusion through its internal faces. Boundary
    coefficients continue to use the adjacent cell's value.

    Parameters
    ----------
    gamma : torch.Tensor | float
        Diffusion coefficient (e.g., kinematic viscosity or conductivity).
        If tensor, values are cell-centered with size ``num_cells``.
        If float, a uniform value is used for all cells.
    field : CellField
        Target field.

    Returns
    -------
    FvMatrix
        Coefficient matrix assembled from diffusion term.
    """
    if isinstance(gamma, torch.Tensor):
        require_shape(gamma, (field.grid.num_cells,), "diffusivity")
    mat = FvMatrix(field)
    grid = field.grid
    geo = face_geometry(grid)
    scheme = search_laplacian_scheme(grid.sim_config, field)

    # Interpolate gamma to face centers.
    harmonic = scheme in (
        LaplacianScheme.GAUSS_HARMONIC_CORRECTED,
        LaplacianScheme.GAUSS_HARMONIC_UNCORRECTED,
    )
    gamma_f = _interpolate_gamma(geo, gamma, harmonic=harmonic)

    # Face diffusion coefficient: gamma * |Sf| / |d . n|
    coeff = gamma_f * geo.mag_Sf_all * geo.delta_coeffs_all

    # Cut immersed split faces for dual-sided IBM treatment.
    if isinstance(grid, AxisProjectedGrid) and grid.num_immersed_faces > 0:
        coeff = coeff.masked_fill(grid.ap_is_immersed_faces, 0.0)

    mat.upper = coeff
    mat.lower = coeff

    # Subtract from owner and neighbor diagonal contributions.
    mat.diag.index_add_(0, geo.owner, -coeff)
    mat.diag.index_add_(0, geo.neighbour, -coeff)

    # Skewness correction on hanging-node faces (explicit, deferred).
    if (
        scheme
        in (
            LaplacianScheme.CORRECTED,
            LaplacianScheme.GAUSS_HARMONIC_CORRECTED,
        )
        and geo.num_hanging > 0
    ):
        correction_src = _hanging_correction_source(field, geo, gamma_f)
        hang = geo.hang_idx
        mat.source.index_add_(0, geo.owner_s[hang], -correction_src)
        mat.source.index_add_(0, geo.neighbour_s[hang], correction_src)

        # Store the explicit correction so that ``FvMatrix.flux`` reproduces
        # the full discrete face flux (OpenFOAM ``faceFluxCorrectionPtr``).
        face_flux_correction = torch.zeros(
            (geo.num_single, *field.component_shape),
            dtype=grid.dtype,
            device=grid.device,
        )
        face_flux_correction.index_add_(0, hang, correction_src)
        mat.face_flux_correction = face_flux_correction

    for batch in iter_boundary_batches(field):
        f, ref_v, ref_g, _ = evaluate_boundary_state(field, batch)
        gamma_bnd = (
            gamma[batch.target_cells]
            if isinstance(gamma, torch.Tensor)
            else gamma
        )
        area = torch.linalg.vector_norm(outward_boundary_Sf(grid, batch), dim=1)
        conductance = gamma_bnd * area
        # (u_b - u_P) / d already includes the ghost reconstruction's 1/theta.
        # Constrain near Dirichlet cells after assembling the full equation.
        c_value = conductance * boundary_value_gradient_coefficient(
            field, batch, f
        )
        c_gradient = conductance * (1.0 - f)
        src = (
            broadcast_entity(c_value, ref_v) * ref_v
            + broadcast_entity(c_gradient, ref_g) * ref_g
        )
        mat.diag.index_add_(0, batch.target_cells, -c_value)
        mat.source.index_add_(0, batch.target_cells, -src)

    return mat

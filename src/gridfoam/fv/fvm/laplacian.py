import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
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
    if scheme == LaplacianScheme.LINEAR:
        return LaplacianScheme.CORRECTED
    return scheme


def _interpolate_gamma(
    geo: FaceGeometry, gamma: Float[torch.Tensor, " C 1"] | float
) -> Float[torch.Tensor, " F 1"] | float:
    """Uncorrected linear interpolation of ``gamma`` to all internal faces."""
    if not isinstance(gamma, torch.Tensor):
        return gamma
    # All internal faces, including immersed; those coefficients are zeroed
    # later so the LDU arrays keep internal-face length.
    return (
        geo.w_all * gamma[geo.owner] + (1.0 - geo.w_all) * gamma[geo.neighbour]
    )


def _hanging_correction_source(
    field: CellField,
    geo: FaceGeometry,
    gamma_f: Float[torch.Tensor, " F 1"] | float,
) -> Float[torch.Tensor, " F_hang k"]:
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
    return gamma_hang * geo.mag_Sf_s[geo.hang_idx] * correction


def laplacian(
    gamma: Float[torch.Tensor, " C 1"] | float, field: CellField
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
    mat = FvMatrix(field)
    grid = field.grid
    geo = face_geometry(grid)
    scheme = search_laplacian_scheme(grid.sim_config, field)

    # Interpolate gamma to face centers.
    gamma_f = _interpolate_gamma(geo, gamma)

    # Face diffusion coefficient: gamma * |Sf| / |d . n|
    coeff = gamma_f * geo.mag_Sf_all * geo.delta_coeffs_all

    # Cut immersed split faces for dual-sided IBM treatment.
    if isinstance(grid, AxisProjectedGrid) and grid.num_immersed_faces > 0:
        coeff = coeff.masked_fill(grid.ap_is_immersed_faces[:, None], 0.0)

    mat.upper = coeff
    mat.lower = coeff

    # Subtract from owner and neighbor diagonal contributions.
    mat.diag.index_add_(0, geo.owner, -coeff)
    mat.diag.index_add_(0, geo.neighbour, -coeff)

    # Skewness correction on hanging-node faces (explicit, deferred).
    if scheme == LaplacianScheme.CORRECTED and geo.num_hanging > 0:
        correction_src = _hanging_correction_source(field, geo, gamma_f)
        hang = geo.hang_idx
        mat.source.index_add_(0, geo.owner_s[hang], -correction_src)
        mat.source.index_add_(0, geo.neighbour_s[hang], correction_src)

        # Store the explicit correction so that ``FvMatrix.flux`` reproduces
        # the full discrete face flux (OpenFOAM ``faceFluxCorrectionPtr``).
        face_flux_correction = torch.zeros(
            (geo.num_single, field.num_components),
            dtype=grid.dtype,
            device=grid.device,
        )
        face_flux_correction.index_add_(0, hang, correction_src)
        mat.face_flux_correction = face_flux_correction

    # Domain boundaries
    for batch in iter_boundary_batches(field):
        f, ref_v, ref_g, _ = evaluate_boundary_state(field, batch)
        gamma_bnd = (
            gamma[batch.target_cells]
            if isinstance(gamma, torch.Tensor)
            else gamma
        )
        # Domain boundaries
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            mag_Sf_bnd = torch.linalg.vector_norm(
                grid.domain_bnd_Sf[batch.face_mask], dim=1, keepdim=True
            )
            c_dirichlet = gamma_bnd * mag_Sf_bnd / batch.mag_d
            c_neumann = gamma_bnd * mag_Sf_bnd
            diag = -f * c_dirichlet
            src = f * c_dirichlet * ref_v + (1.0 - f) * c_neumann * ref_g
            mat.diag.index_add_(0, batch.target_cells, diag)
            mat.source.index_add_(0, batch.target_cells, -src)
            continue

        # Immersed boundaries
        if isinstance(grid, AxisProjectedGrid):
            immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
            mag_Sf_bnd = torch.linalg.vector_norm(
                immersed_Sf[batch.face_mask], dim=1, keepdim=True
            )
            c_dirichlet = gamma_bnd * mag_Sf_bnd / batch.mag_d
            c_neumann = gamma_bnd * mag_Sf_bnd
            if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
                wb = grid.ap_owner_weights[batch.face_mask, 0:1]
                w = grid.ap_owner_weights[batch.face_mask, 1:2]

                diag = c_dirichlet * (w - 1.0 + wb * (1.0 - f))
                src = wb * (
                    f * c_dirichlet * ref_v + (1.0 - f) * c_neumann * ref_g
                )
                mat.diag.index_add_(0, batch.target_cells, diag)
                mat.source.index_add_(0, batch.target_cells, -src)
                continue

            elif batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                wb = grid.ap_neighbour_weights[batch.face_mask, 0:1]
                w = grid.ap_neighbour_weights[batch.face_mask, 1:2]

                diag = c_dirichlet * (w - 1.0 + wb * (1.0 - f))
                src = wb * (
                    f * c_dirichlet * ref_v + (1.0 - f) * c_neumann * ref_g
                )
                mat.diag.index_add_(0, batch.target_cells, diag)
                mat.source.index_add_(0, batch.target_cells, -src)
                continue

    return mat

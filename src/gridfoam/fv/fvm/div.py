from gridfoam.core.dimensions import DIM_VOL_FLUX, assert_compatible
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.shapes import broadcast_entity
from gridfoam.fv.boundary_ops import (
    boundary_block,
    evaluate_boundary_state,
    iter_boundary_batches,
)
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.fv.schemes.div import get_div_scheme
from gridfoam.fv.schemes.selection import search_div_scheme


def div(phi: FaceField, field: CellField) -> FvMatrix:
    """
    Build the convection (divergence) matrix term.

    Represents div(U * psi). Boundary values follow the prescribed boundary
    condition for either flux direction; inlet/outlet switching belongs to
    the boundary condition itself.

    Parameters
    ----------
    phi : FaceField
        Face mass/volumetric flux field.
    field : CellField
        Advected cell-centered field.

    Returns
    -------
    FvMatrix
        Coefficient matrix assembled from convection term.

    Raises
    ------
    DimensionMismatchError
        If ``phi`` does not carry a volumetric flux dimension.
    """
    if phi.component_shape != ():
        raise ValueError("convection requires scalar face flux")
    mat = FvMatrix(field)
    grid = field.grid
    assert_compatible(phi.dimension, DIM_VOL_FLUX, "fvm.div flux field")
    div_scheme = search_div_scheme(grid.sim_config, phi, field)

    # Internal faces
    scheme_func = get_div_scheme(div_scheme)
    upper, lower, diag_O, diag_N, source_face = scheme_func(phi, field)

    geo = face_geometry(grid)
    mat.upper.index_copy_(0, geo.single_idx, upper)
    mat.lower.index_copy_(0, geo.single_idx, lower)
    mat.diag.index_add_(0, geo.owner_s, diag_O)
    mat.diag.index_add_(0, geo.neighbour_s, diag_N)
    mat.source.index_add_(0, geo.owner_s, source_face)
    mat.source.index_add_(0, geo.neighbour_s, -source_face)

    for batch in iter_boundary_batches(field):
        f, ref_v, ref_g, _ = evaluate_boundary_state(field, batch)
        f_view = broadcast_entity(f, ref_v)
        distance = broadcast_entity(batch.mag_d, ref_g)
        boundary_source = f_view * ref_v + (1.0 - f_view) * distance * ref_g

        # psi_b = (1 - f) * psi_P + boundary_source. The boundary condition
        # selects f, including any flow-direction switching in inletOutlet.
        flux = boundary_block(phi, batch.face_kind)[batch.face_mask]
        diag = flux * (1.0 - f)
        src = broadcast_entity(flux, boundary_source) * boundary_source
        mat.diag.index_add_(0, batch.target_cells, diag)
        mat.source.index_add_(0, batch.target_cells, -src)

    return mat

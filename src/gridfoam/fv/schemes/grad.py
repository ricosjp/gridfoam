"""Cell-centered gradient schemes."""

from collections.abc import Callable

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.fv.kernels.gauss_gradient import assemble_gauss_gradient
from gridfoam.fv.kernels.least_squares import least_squares_gradient
from gridfoam.fv.schemes.interpolate import linear as interpolate_linear
from gridfoam.fv.schemes.selection import search_grad_scheme
from gridfoam.meta.enums import GradScheme

GradSchemeFunc = Callable[
    [CellField], Float[torch.Tensor, " C *component_shape 3"]
]


def linear(field: CellField) -> Float[torch.Tensor, " C *component_shape 3"]:
    """
    Green-Gauss gradient from hierarchy-aware linear face values.

    Face values come from :func:`gridfoam.fv.schemes.interpolate.linear`,
    whose hanging-face correction uses a local least-squares gradient, so
    the result is exact for linear fields on 2:1 interfaces. It costs one
    face interpolation plus one Green-Gauss assembly and is mainly useful
    when the interpolated face field is needed anyway.

    Parameters
    ----------
    field : CellField
        Cell-centered field.

    Returns
    -------
    torch.Tensor
        Cell-centered gradient with shape ``[C, *component_shape, 3]``.
    """
    geo = face_geometry(field.grid)
    psi_f = interpolate_linear(field)
    return assemble_gauss_gradient(field.grid, psi_f, geo)


def leastsquare(
    field: CellField,
) -> Float[torch.Tensor, " C *component_shape 3"]:
    """
    Weighted least-squares gradient.

    Internal equations use only single-sided faces. Immersed faces are
    never mixed into the owner/neighbour stencil; their upper/lower
    boundary values enter as independent boundary equations. The normal
    matrix is geometric and cached on the grid, so each evaluation is one
    scatter of the right-hand side and one batched ``3x3`` product.
    Exact for linear fields on every cell of an octree grid.

    Parameters
    ----------
    field : CellField
        Cell-centered field.

    Returns
    -------
    torch.Tensor
        Cell-centered gradient with shape ``[C, *component_shape, 3]``.
    """
    return least_squares_gradient(field)


GRAD_SCHEMES: dict[GradScheme, GradSchemeFunc] = {
    GradScheme.LINEAR: linear,
    GradScheme.LEASTSQUARE: leastsquare,
}


def get_grad_scheme(scheme: GradScheme) -> GradSchemeFunc:
    """Return the gradient-scheme function for the given enum."""
    return GRAD_SCHEMES[scheme]


def eval_grad(field: CellField) -> Float[torch.Tensor, " C *component_shape 3"]:
    """
    Evaluate the configured cell-centered gradient as a tensor.

    Schemes that need a gradient must call this helper rather than
    ``fvc.grad``, which wraps the same tensor in a named ``CellField``.

    Parameters
    ----------
    field : CellField
        Target cell-centered field.

    Returns
    -------
    torch.Tensor
        Cell-centered gradient with shape ``[C, *component_shape, 3]``.
    """
    scheme = search_grad_scheme(field.grid.sim_config, field)
    return get_grad_scheme(scheme)(field)

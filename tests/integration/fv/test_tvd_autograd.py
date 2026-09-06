"""TVD corrections must remain differentiable on constant and flat regions."""

import pytest
import torch
from tests.helpers import refined_grid

from gridfoam.core.field import CellField, FaceField
from gridfoam.fv.schemes.div import get_div_scheme
from gridfoam.meta.enums import DivScheme, FieldRole


@pytest.mark.parametrize("num_components", [1, 3])
@pytest.mark.parametrize("plateau", [False, True])
@pytest.mark.parametrize(
    "scheme",
    [s for s in DivScheme if s not in (DivScheme.UPWIND, DivScheme.LINEAR)],
)
def test_tvd_backward_on_flat_regions(
    scheme: DivScheme, plateau: bool, num_components: int
):
    grid = refined_grid()
    field = CellField(grid, "psi", FieldRole.LOCAL, num_components)
    values = torch.ones_like(field.data)
    if plateau:
        x = grid.cell_centers[:, :1]
        values = values + torch.clamp(x - x.mean(), min=0.0)
    values.requires_grad_()
    field.data = values
    phi = FaceField(grid, "phi", FieldRole.LOCAL, 1)
    # Exercise both owner- and neighbour-upwind branches.
    face_idx = torch.arange(phi.num_single_sided, device=grid.device)
    phi.single_data = torch.where(face_idx[:, None] % 2 == 0, 1.0, -1.0).to(
        grid.dtype
    )

    source = get_div_scheme(scheme)(phi, field)[-1]
    assert torch.isfinite(source).all()
    if not plateau:
        torch.testing.assert_close(source, torch.zeros_like(source))
    # Do not sum assembled cell sources: conservation would cancel them.
    weights = torch.linspace(
        0.5, 1.5, source.numel(), dtype=grid.dtype, device=grid.device
    ).reshape_as(source)
    (gradient,) = torch.autograd.grad((source * weights).sum(), values)
    assert torch.isfinite(gradient).all()
    if not plateau:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))

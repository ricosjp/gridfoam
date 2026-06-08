import torch
from tests.conftest import small_gridfoam_config

from gridfoam.core.field import CellField
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvc
from gridfoam.meta.config import RefinementRegionConfig
from gridfoam.meta.enums import FieldRole


def _refined_grid() -> IGridBase:
    config = small_gridfoam_config()
    config = config.model_copy(
        update={
            "fluxel": config.fluxel.model_copy(
                update={
                    "root_resolution": [4, 4, 1],
                    "refinement_regions": [
                        RefinementRegionConfig(
                            name="center",
                            min=[0.25, 0.25, 0.0],
                            max=[0.75, 0.75, 0.1],
                            level=1,
                        )
                    ],
                }
            )
        }
    )
    return create_grid(config)


def _linear_scalar_field(
    grid: IGridBase,
) -> tuple[CellField, torch.Tensor]:
    field = CellField(
        grid,
        name="linear_scalar",
        role=FieldRole.LOCAL,
        num_components=1,
        export=False,
    )
    gradient = torch.tensor(
        [2.0, -3.0, 5.0], dtype=grid.dtype, device=grid.device
    )
    field.data = (grid.cell_centers @ gradient + 7.0).reshape(-1, 1)
    return field, gradient


def test_interpolate_is_linear_exact_on_octree_interfaces():
    grid = _refined_grid()
    field, gradient = _linear_scalar_field(grid)

    interpolated = fvc.interpolate(field)
    expected = (
        grid.face_centers[interpolated.single_mask] @ gradient + 7.0
    ).reshape(-1, 1)

    torch.testing.assert_close(
        interpolated.single_data, expected, atol=1e-12, rtol=1e-12
    )


def test_sn_grad_removes_octree_interface_skewness_for_linear_fields():
    grid = _refined_grid()
    field, gradient = _linear_scalar_field(grid)

    sn_grad = fvc.sn_grad(field)
    expected = gradient[grid.axis[sn_grad.single_mask]].reshape(-1, 1)

    torch.testing.assert_close(
        sn_grad.single_data, expected, atol=1e-12, rtol=1e-12
    )

import pytest
import torch

from gridfoam.settings import CubeSetting, GridSetting


@pytest.fixture
def grid_settings() -> GridSetting:
    return GridSetting(
        blockXMin=0.0,
        blockXMax=2.0,
        blockYMin=0.0,
        blockYMax=1.0,
        blockZMin=0.0,
        blockZMax=1.0,
        nBlockX=4,
        nBlockY=2,
        nBlockZ=2,
    )


def test_get_domain(grid_settings: GridSetting):
    domain = grid_settings.get_domain()
    torch.testing.assert_close(domain.min, torch.tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(domain.max, torch.tensor([2.0, 1.0, 1.0]))


def test_get_block_resolutions(grid_settings: GridSetting):
    divisions = grid_settings.get_block_resolutions()
    torch.testing.assert_close(
        divisions, torch.tensor([4, 2, 2], dtype=torch.int32)
    )


@pytest.fixture
def cube_settings() -> CubeSetting:
    return CubeSetting(
        nx=8,
        ny=8,
        nz=8,
        n_bnd_x=2,
        n_bnd_y=2,
        n_bnd_z=2,
    )


def test_properties(cube_settings: CubeSetting):
    assert cube_settings.tnx == 12
    assert cube_settings.tny == 12
    assert cube_settings.tnz == 12
    torch.testing.assert_close(
        cube_settings.res, torch.tensor([8, 8, 8], dtype=torch.int32)
    )
    torch.testing.assert_close(
        cube_settings.bnd_width, torch.tensor([2, 2, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        cube_settings.tres, torch.tensor([12, 12, 12], dtype=torch.int32)
    )

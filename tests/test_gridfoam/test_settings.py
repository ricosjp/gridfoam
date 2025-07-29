import pytest
import torch

from gridfoam.settings import GridSetting


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


def test_properties(grid_settings: GridSetting):
    assert grid_settings.cube_setting.width == 8
    assert grid_settings.cube_setting.bnd_width == 2

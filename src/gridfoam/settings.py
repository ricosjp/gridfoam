from typing import Literal

import torch
from jaxtyping import Int32
from pydantic import BaseModel, Field

from gridfoam._geometry import AABB
from gridfoam.utils.enums import Constants


class CubeSetting(BaseModel, frozen=True):
    width: int = Field(default=8, ge=2, le=16)
    bnd_width: Literal[1, 2] = Field(default=2)

class GridSetting(BaseModel, frozen=True):
    blockXMin: float
    blockXMax: float
    blockYMin: float
    blockYMax: float
    blockZMin: float = Field(default=0.0)
    blockZMax: float = Field(default=1.0)
    nBlockX: int
    nBlockY: int
    nBlockZ: int = Field(default=1)
    alpha: float = Field(default=0.3, le=0.5)
    level_limit: int = Field(default=5, ge=1, le=Constants.MAX_LEVEL)
    cube_setting: CubeSetting = Field(default_factory=CubeSetting)

    def get_domain(self) -> AABB:
        min_pt = torch.tensor([self.blockXMin, self.blockYMin, self.blockZMin])
        max_pt = torch.tensor([self.blockXMax, self.blockYMax, self.blockZMax])
        return AABB(min_pt, max_pt)

    def get_block_resolutions(self) -> Int32[torch.Tensor, " 3"]:
        return torch.tensor(
            [self.nBlockX, self.nBlockY, self.nBlockZ], dtype=torch.int32
        )

from typing import Literal

import torch
from jaxtyping import Int32
from pydantic import BaseModel, Field, computed_field

from gridfoam._geometry import AABB
from gridfoam.utils.enums import Constants


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

    def get_domain(self) -> AABB:
        min_pt = torch.tensor([self.blockXMin, self.blockYMin, self.blockZMin])
        max_pt = torch.tensor([self.blockXMax, self.blockYMax, self.blockZMax])
        return AABB(min_pt, max_pt)

    def get_block_resolutions(self) -> Int32[torch.Tensor, " 3"]:
        return torch.tensor(
            [self.nBlockX, self.nBlockY, self.nBlockZ], dtype=torch.int32
        )


class CubeSetting(BaseModel, frozen=True):
    nx: int = Field(default=8, ge=1, le=16)
    ny: int = Field(default=8, ge=1, le=16)
    nz: int = Field(default=8, ge=1, le=16)
    n_bnd_x: Literal[1, 2] = Field(default=2)
    n_bnd_y: Literal[1, 2] = Field(default=2)
    n_bnd_z: Literal[1, 2] = Field(default=2)

    @computed_field
    @property
    def tnx(self) -> int:
        return self.nx + 2 * self.n_bnd_x

    @computed_field
    @property
    def tny(self) -> int:
        return self.ny + 2 * self.n_bnd_y

    @computed_field
    @property
    def tnz(self) -> int:
        return self.nz + 2 * self.n_bnd_z

    @property
    def res(self) -> Int32[torch.Tensor, " 3"]:
        return torch.tensor([self.nx, self.ny, self.nz], dtype=torch.int32)

    @property
    def bnd_width(self) -> Int32[torch.Tensor, " 3"]:
        return torch.tensor(
            [self.n_bnd_x, self.n_bnd_y, self.n_bnd_z], dtype=torch.int32
        )

    @property
    def tres(self) -> Int32[torch.Tensor, " 3"]:
        return torch.tensor([self.tnx, self.tny, self.tnz], dtype=torch.int32)

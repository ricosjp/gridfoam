import re
from typing import Annotated, Literal

import torch
from jaxtyping import Int32
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.functional_validators import PlainValidator

from gridfoam._geometry import AABB
from gridfoam.utils.enums import Constants


def device_validator(v: str | torch.device) -> torch.device:
    if isinstance(v, torch.device):
        return v
    if v == "cpu":
        return torch.device("cpu")
    if re.fullmatch(r"cuda:\d+", v):
        return torch.device(v)
    raise ValueError("device must be 'cpu' or 'cuda:<int>' (e.g. 'cuda:0')")


def dtype_validator(v: str | torch.dtype) -> torch.dtype:
    if isinstance(v, torch.dtype):
        return v
    try:
        return getattr(torch, v)
    except AttributeError as e:
        raise ValueError(f"Invalid dtype string: {v}") from e


TorchDevice = Annotated[torch.device, PlainValidator(device_validator)]
TorchDtype = Annotated[torch.dtype, PlainValidator(dtype_validator)]


class FieldDataAttribute(BaseModel, frozen=True):
    """Field data information.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the field data.
        e.g. (3,) for velocity, (1,) for pressure
    dtype : str
        Data type of the field data.
        e.g. "bool" / "int32" / "int64" / "float32" / "float64"
    """

    shape: tuple[int, ...]
    dtype: TorchDtype

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(x <= 0 for x in v):
            raise ValueError("All elements must be > 0")
        return v


class CubeSetting(BaseModel, frozen=True):
    width: int = Field(default=8, ge=8, le=16)
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
    """
    alpha : float
        Refinement factor. This value is used
        to determine if a node should be split
        based on the cube size and mesh curvature.
    """

    depth_limit: int = Field(default=5, ge=1, le=Constants.MAX_OCTREE_DEPTH)
    """
    depth_limit : int
        Maximum depth of the octree.
        This value is used to limit the depth of the octree.
    """

    cube_setting: CubeSetting = Field(default_factory=CubeSetting)
    """
    cube_setting : CubeSetting
        Cube setting.
        This value is used to determine the size of the cube.
    """

    field_data_dict: dict[str, FieldDataAttribute] = Field(default_factory=dict)
    """Field data setting.

    Parameters
    ----------
    field_data_dict : dict[str, FieldDataAttribute]
        Field data dictionary.
        e.g. {
            "U": FieldDataAttribute(shape=(3,), dtype=torch.float32),
            "p": FieldDataAttribute(shape=(1,), dtype=torch.float32)
        }
    """

    device: TorchDevice = Field(default=torch.device("cpu"))
    """
    device : torch.device
        Device on which tensors are allocated.
    """

    @model_validator(mode="before")
    @classmethod
    def fill_missing_keys(
        cls, v: dict
    ) -> dict:
        field_data_dict: dict = v.get("field_data_dict", {})
        field_data_dict.setdefault("U", FieldDataAttribute(shape=(3,), dtype=torch.float32))
        field_data_dict.setdefault("p", FieldDataAttribute(shape=(1,), dtype=torch.float32))
        v["field_data_dict"] = field_data_dict
        return v

    def get_domain(self) -> AABB:
        min_pt = torch.tensor([self.blockXMin, self.blockYMin, self.blockZMin])
        max_pt = torch.tensor([self.blockXMax, self.blockYMax, self.blockZMax])
        return AABB(min_pt, max_pt)

    def get_block_resolutions(self) -> Int32[torch.Tensor, " 3"]:
        return torch.tensor(
            [self.nBlockX, self.nBlockY, self.nBlockZ], dtype=torch.int32
        )

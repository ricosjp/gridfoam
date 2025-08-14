from dataclasses import dataclass

import torch
from jaxtyping import Int32

from gridfoam._base.cube import Cube
from gridfoam.utils.annotated_type import CubeCode
from gridfoam.utils.enums import CubeType


@dataclass(kw_only=True)
class LeafCube(Cube):
    face_ids: Int32[torch.Tensor, " n_faces"]
    neighbor_codes: list[CubeCode]
    cube_type: CubeType = CubeType.LEAF


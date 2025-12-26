from dataclasses import dataclass

import torch
from jaxtyping import Float, UInt64

from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta


@dataclass
class CtxForCubeOperation:
    depth: int
    bounds: UInt64[torch.Tensor, "3"]

    dt: float
    dx: Float[torch.Tensor, " 3"]

    vertices: Float[torch.Tensor, "n_vertices 3"]
    bcs: list[BoundaryConditionMeta]

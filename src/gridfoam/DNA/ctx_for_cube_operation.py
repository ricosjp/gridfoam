from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, UInt64

from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta


@dataclass
class CtxForCubeOperation:
    depth: int
    bounds: UInt64[np.ndarray, "3"]

    dt: float
    dx: Float[torch.Tensor, " 3"]

    vertices: Float[torch.Tensor, "n_vertices 3"]
    bcs: list[BoundaryConditionMeta]

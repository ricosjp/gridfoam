from dataclasses import dataclass, field

import torch
from jaxtyping import Float

from gridfoam.DNA.enum import BoundaryConditionType
from gridfoam.DNA.meta.field import FieldMeta


@dataclass(slots=True)
class BoundaryConditionMeta:
    # Identification
    name: str                     # e.g. "inlet", "outlet", "wall"

    # Boundary condition
    # "domainX+", "domainX-", "domainY+", "domainY-", "domainZ+", "domainZ-"
    # are the built-in boundary labels that represent the faces of the domain.
    target_field: FieldMeta
    type: BoundaryConditionType
    value: Float[torch.Tensor, " C"]
    target_boundary_labels: list[str] = field(default_factory=list)

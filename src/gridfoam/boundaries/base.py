from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from gridfoam.meta.enums import BoundaryConditionType, FaceSide
from gridfoam.meta.types import PatchName

if TYPE_CHECKING:
    from gridfoam.core.field import CellField


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary condition evaluation.

    Any boundary condition is reduced to three tensors:
    blend fraction, reference value, and reference gradient.
    """

    @property
    @abstractmethod
    def type(self) -> BoundaryConditionType:
        pass

    @abstractmethod
    def component(self, c: int) -> BoundaryCondition:
        pass

    @abstractmethod
    def evaluate(
        self,
        field: CellField,
        patch_name: PatchName,
        side: FaceSide = FaceSide.UPPER,
    ) -> tuple[
        Float[torch.Tensor, " F_patch 1"],
        Float[torch.Tensor, " F_patch k"],
        Float[torch.Tensor, " F_patch k"],
    ]:
        """
        Evaluate boundary condition values in value-fraction form.

        Parameters
        ----------
        field : CellField
            Field to which the boundary condition is applied.
        patch_name : PatchName
            Target patch name.
        side : FaceSide, optional
            Face side for immersed-boundary evaluation.
            The default is ``FaceSide.UPPER``.

        Returns
        -------
        tuple[
            Float[torch.Tensor, " F_patch 1"],
            Float[torch.Tensor, " F_patch k"],
            Float[torch.Tensor, " F_patch k"],
        ]
            Tuple ``(fraction, ref_value, ref_grad)``.
            - ``fraction``: [F_patch, 1], 1.0 for Dirichlet
              and 0.0 for Neumann.
            - ``ref_value``: [F_patch, k], fixed value
              for Dirichlet contribution.
            - ``ref_grad``: [F_patch, k], fixed normal gradient
              for Neumann contribution.
        """
        pass

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from jaxtyping import Float

from gridfoam.meta.enums import BoundaryConditionType, FaceSide
from gridfoam.meta.types import PatchName

if TYPE_CHECKING:
    from gridfoam.core.field import CellField, GeometricField
else:
    CellField = Any
    GeometricField = Any


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary condition evaluation.

    Any boundary condition is reduced to three tensors:
    blend fraction, reference value, and reference gradient.

    Attributes
    ----------
    type : BoundaryConditionType
        Serialized boundary-condition type.
    """

    @property
    @abstractmethod
    def type(self) -> BoundaryConditionType:
        """Serialized boundary-condition type."""
        pass

    @property
    def assignable(self) -> bool:
        """
        Whether the boundary value may be overwritten by the solution.

        Mirrors OpenFOAM ``fvPatchField::assignable``. Fixed-value style
        conditions return ``False`` so that ``constrainHbyA`` copies the
        velocity boundary value into ``HbyA``; gradient-type and
        inlet-outlet conditions return ``True``.
        """
        return True

    def dependencies(self, field: CellField) -> tuple[GeometricField, ...]:
        """
        Fields other than ``field`` that :meth:`evaluate` reads.

        Used to invalidate cached boundary states. Conditions that only
        depend on ``field`` and static data return an empty tuple.

        Parameters
        ----------
        field : CellField
            Field the condition is attached to.

        Returns
        -------
        tuple[GeometricField, ...]
            Registered fields whose data influences the evaluation.
        """
        del field
        return ()

    @abstractmethod
    def evaluate(
        self,
        field: CellField,
        patch_name: PatchName,
        side: FaceSide = FaceSide.UPPER,
    ) -> tuple[
        Float[torch.Tensor, " F_patch"],
        Float[torch.Tensor, " F_patch *component_shape"],
        Float[torch.Tensor, " F_patch *component_shape"],
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
            Float[torch.Tensor, " F_patch"],
            Float[torch.Tensor, " F_patch *component_shape"],
            Float[torch.Tensor, " F_patch *component_shape"],
        ]
        - ``fraction``: ``[F_patch]``, 1.0 for Dirichlet
            and 0.0 for Neumann.
        - ``ref_value``: ``[F_patch, *component_shape]``, fixed value
            for Dirichlet contribution.
        - ``ref_grad``: ``[F_patch, *component_shape]``, fixed normal gradient
            for Neumann contribution.
        """
        pass

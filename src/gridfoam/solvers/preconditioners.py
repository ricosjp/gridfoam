from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from jaxtyping import Float

from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.shapes import broadcast_entity
from gridfoam.meta.enums import PreconditionerType


def create_preconditioner(
    precon_type: PreconditionerType, A: FvMatrix
) -> Preconditioner:
    match precon_type:
        case PreconditionerType.NONE:
            return NonePreconditioner(A)
        case PreconditionerType.JACOBI:
            return JacobiPreconditioner(A)
        case _:
            raise ValueError(f"Invalid preconditioner type: {precon_type}")


class Preconditioner(ABC):
    """
    Base class for linear-system preconditioners.

    Parameters
    ----------
    A : FvMatrix
        Matrix to precondition.
    """

    def __init__(self, A: FvMatrix):
        self.A = A
        self.setup()

    @abstractmethod
    def setup(self):
        """Build preconditioner state from matrix ``A``."""
        pass

    @abstractmethod
    def apply(
        self, r: Float[torch.Tensor, " C *component_shape"]
    ) -> Float[torch.Tensor, " C *component_shape"]:
        """
        Apply preconditioning and return ``z = M^{-1} r``.

        Parameters
        ----------
        r : torch.Tensor
            Residual vector with shape ``[C, *component_shape]``.

        Returns
        -------
        torch.Tensor
            Preconditioned vector.
        """
        pass


class NonePreconditioner(Preconditioner):
    """Identity preconditioner (no-op, ``M = I``)."""

    def setup(self):
        pass

    def apply(
        self, r: Float[torch.Tensor, " C *component_shape"]
    ) -> Float[torch.Tensor, " C *component_shape"]:
        return r


class JacobiPreconditioner(Preconditioner):
    """
    Jacobi (diagonal) preconditioner.

    Applies inverse diagonal scaling and supports multi-component fields.
    """

    def setup(self):
        dtype = self.A.diag.dtype
        vsmall = torch.finfo(dtype).eps

        diag_abs = torch.abs(self.A.diag)
        self.inv_diag = torch.where(
            diag_abs < vsmall, torch.ones_like(self.A.diag), 1.0 / self.A.diag
        )

    def apply(
        self, r: Float[torch.Tensor, " C *component_shape"]
    ) -> Float[torch.Tensor, " C *component_shape"]:
        inv_diag = broadcast_entity(self.inv_diag, r)
        return inv_diag * r

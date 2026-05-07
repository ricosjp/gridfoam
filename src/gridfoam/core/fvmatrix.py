from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.grid.base import IGridBase


class FvMatrix:
    """
    Discretized finite-volume matrix equation container (Ax = b).

    Parameters
    ----------
    field : CellField
        Target field solved by this matrix equation.
    """

    def __init__(self, field: CellField):
        self._field = field
        self._grid = field.grid
        dtype = field.grid.dtype
        device = field.grid.device
        n_cells = self.grid.num_cells
        n_faces = self.grid.num_internal_faces
        k = field.num_components

        # LDU
        self._diag = torch.zeros((n_cells, 1), dtype=dtype, device=device)
        self._upper = torch.zeros((n_faces, 1), dtype=dtype, device=device)
        self._lower = torch.zeros((n_faces, 1), dtype=dtype, device=device)
        self._source = torch.zeros((n_cells, k), dtype=dtype, device=device)

    @property
    def field(self) -> CellField:
        return self._field

    @property
    def grid(self) -> IGridBase:
        return self._grid

    @property
    def num_components(self) -> int:
        return self.field.num_components

    @property
    def diag(self) -> Float[torch.Tensor, " C 1"]:
        """Return diagonal coefficients."""
        return self._diag

    @diag.setter
    def diag(self, value: Float[torch.Tensor, " C 1"]):
        """Set diagonal coefficients."""
        self._diag = value

    @property
    def upper(self) -> Float[torch.Tensor, " F 1"]:
        """Return upper off-diagonal coefficients."""
        return self._upper

    @upper.setter
    def upper(self, value: Float[torch.Tensor, " F 1"]):
        """Set upper off-diagonal coefficients."""
        self._upper = value

    @property
    def lower(self) -> Float[torch.Tensor, " F 1"]:
        """Return lower off-diagonal coefficients."""
        return self._lower

    @lower.setter
    def lower(self, value: Float[torch.Tensor, " F 1"]):
        """Set lower off-diagonal coefficients."""
        self._lower = value

    @property
    def source(self) -> Float[torch.Tensor, " C k"]:
        """Return right-hand-side source term."""
        return self._source

    @source.setter
    def source(self, value: Float[torch.Tensor, " C k"]):
        """Set right-hand-side source term."""
        self._source = value

    def __add__(self, other: FvMatrix) -> FvMatrix:
        res = FvMatrix(self.field)
        res.diag = self.diag + other.diag
        res.upper = self.upper + other.upper
        res.lower = self.lower + other.lower
        res.source = self.source + other.source
        return res

    def __sub__(self, other: FvMatrix) -> FvMatrix:
        res = FvMatrix(self.field)
        res.diag = self.diag - other.diag
        res.upper = self.upper - other.upper
        res.lower = self.lower - other.lower
        res.source = self.source - other.source
        return res

    def __neg__(self) -> FvMatrix:
        res = FvMatrix(self.field)
        res.diag = -self.diag
        res.upper = -self.upper
        res.lower = -self.lower
        res.source = -self.source
        return res

    def A(self) -> Float[torch.Tensor, " C 1"]:
        """
        OpenFOAM-style ``A()`` operator.

        Returns volume-normalized diagonal entries: ``diag / V``.
        """
        return self.diag / self.grid.cell_volumes

    def H(self, x: Float[torch.Tensor, " C k"]) -> Float[torch.Tensor, " C k"]:
        """
        OpenFOAM-style ``H()`` operator.

        Computes and returns:
        ``H = (b - (A - diag) * x) / V = (b - A * x + diag * x) / V``.

        Parameters
        ----------
        x : torch.Tensor
            Current field values with shape ``[C, k]``.

        Returns
        -------
        torch.Tensor
            Computed H values with shape ``[C, k]``.
        """
        res = (
            self.source - self.multiply(x) + self.diag * x
        ) / self.grid.cell_volumes
        return res

    def multiply(
        self, x: Float[torch.Tensor, " C k"]
    ) -> Float[torch.Tensor, " C k"]:
        """
        Compute matrix-vector product ``A * x``.

        Parameters
        ----------
        x : torch.Tensor
            Input vector with shape ``[C, k]``.

        Returns
        -------
        torch.Tensor
            Product vector with shape ``[C, k]``.
        """
        owner = self.grid.owner
        neighbour = self.grid.neighbour

        # 1. Diagonal contribution: A_ii * x_i
        res = self.diag * x

        upper = self.upper * x[neighbour]
        lower = self.lower * x[owner]

        # 2. Upper contribution: A_ij * x_j (i=owner, j=neighbour)
        res.index_add_(0, owner, upper)

        # 3. Lower contribution: A_ji * x_i (i=owner, j=neighbour)
        res.index_add_(0, neighbour, lower)

        return res

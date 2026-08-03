from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase


class FvMatrix:
    """
    Discretized finite-volume matrix equation container (Ax = b).

    Stores LDU coefficients, the source vector, and optional explicit
    face-flux corrections for operators such as the non-orthogonal
    Laplacian. The ``flux(psi)`` method reproduces OpenFOAM
    ``fvMatrix::flux(psi)`` on single-sided internal faces.

    Parameters
    ----------
    field : CellField
        Target field solved by this matrix equation.

    Attributes
    ----------
    field : CellField
        Target field solved by this matrix equation.
    grid : IGridBase
        Computational grid of ``field``.
    num_components : int
        Number of components ``k`` in the target field.
    diag : torch.Tensor
        Diagonal coefficients with shape ``[C, 1]``.
    upper : torch.Tensor
        Upper off-diagonal coefficients with shape ``[F, 1]``.
    lower : torch.Tensor
        Lower off-diagonal coefficients with shape ``[F, 1]``.
    source : torch.Tensor
        Right-hand-side source with shape ``[C, k]``.
    face_flux_correction : torch.Tensor or None
        Optional explicit face-flux correction on single-sided faces.
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

        # Explicit face-flux correction on single-sided internal faces,
        # stored by discretization operators (e.g. non-orthogonal Laplacian).
        # OpenFOAM ``fvMatrix::faceFluxCorrectionPtr`` equivalent.
        self._face_flux_correction: (
            Float[torch.Tensor, " F_single k"] | None
        ) = None

    @property
    def field(self) -> CellField:
        """Target field solved by this matrix equation."""
        return self._field

    @property
    def grid(self) -> IGridBase:
        """Computational grid of ``field``."""
        return self._grid

    @property
    def num_components(self) -> int:
        """Number of components ``k`` in the target field."""
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

    @property
    def face_flux_correction(
        self,
    ) -> Float[torch.Tensor, " F_single k"] | None:
        """
        Explicit face-flux correction on single-sided internal faces.

        Set by discretization operators (e.g. ``laplacian``) so that
        ``flux(psi)`` includes non-orthogonal contributions. Corresponds to
        OpenFOAM ``faceFluxCorrectionPtr``.
        """
        return self._face_flux_correction

    @face_flux_correction.setter
    def face_flux_correction(
        self, value: Float[torch.Tensor, " F_single k"] | None
    ):
        """Set the explicit face-flux correction on single-sided faces."""
        self._face_flux_correction = value

    @staticmethod
    def _combine_face_flux_correction(
        a: Float[torch.Tensor, " F_single k"] | None,
        b: Float[torch.Tensor, " F_single k"] | None,
        sign: float,
    ) -> Float[torch.Tensor, " F_single k"] | None:
        """Combine two optional corrections as ``a + sign * b``."""
        if a is None:
            if b is None:
                return None
            return sign * b
        if b is None:
            return a.clone()
        return a + sign * b

    def __add__(self, other: FvMatrix) -> FvMatrix:
        res = FvMatrix(self.field)
        res.diag = self.diag + other.diag
        res.upper = self.upper + other.upper
        res.lower = self.lower + other.lower
        res.source = self.source + other.source
        res.face_flux_correction = self._combine_face_flux_correction(
            self._face_flux_correction, other._face_flux_correction, 1.0
        )
        return res

    def __sub__(self, other: FvMatrix) -> FvMatrix:
        res = FvMatrix(self.field)
        res.diag = self.diag - other.diag
        res.upper = self.upper - other.upper
        res.lower = self.lower - other.lower
        res.source = self.source - other.source
        res.face_flux_correction = self._combine_face_flux_correction(
            self._face_flux_correction, other._face_flux_correction, -1.0
        )
        return res

    def __neg__(self) -> FvMatrix:
        res = FvMatrix(self.field)
        res.diag = -self.diag
        res.upper = -self.upper
        res.lower = -self.lower
        res.source = -self.source
        if self._face_flux_correction is not None:
            res.face_flux_correction = -self._face_flux_correction
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

    def as_transpose(self) -> FvMatrix:
        """
        Return a shallow view whose :meth:`multiply` computes ``A^T x``.

        Shares storage with ``self`` and swaps ``upper`` / ``lower`` so the
        same Krylov / AMG path can solve the transpose system for the
        implicit adjoint.
        """
        view = FvMatrix.__new__(FvMatrix)
        view._field = self._field
        view._grid = self._grid
        view._diag = self._diag
        view._upper = self._lower
        view._lower = self._upper
        view._source = self._source
        view._face_flux_correction = self._face_flux_correction
        return view

    def _single_internal_mask(self) -> torch.Tensor:
        """Boolean mask selecting single-sided internal faces."""
        grid = self.grid
        mask = torch.ones(
            grid.num_internal_faces, dtype=torch.bool, device=grid.device
        )
        if isinstance(grid, AxisProjectedGrid):
            mask[grid.ap_is_immersed_faces] = False
        return mask

    def flux(
        self, psi: Float[torch.Tensor, " C k"]
    ) -> Float[torch.Tensor, " F_single k"]:
        """
        Face flux from matrix coefficients and ``psi`` field values.

        OpenFOAM ``fvMatrix::flux(psi)`` equivalent on single-sided internal
        faces:

        ``upper * psi[neighbour] - lower * psi[owner]``

        plus the stored explicit face-flux correction (e.g. the non-orthogonal
        Laplacian correction set during matrix assembly), if any.

        Parameters
        ----------
        psi : torch.Tensor
            Field values with shape ``[C, k]``.

        Returns
        -------
        torch.Tensor
            Face flux on single-sided internal faces with shape
            ``[F_single, k]``.
        """
        grid = self.grid
        single_mask = self._single_internal_mask()
        owner = grid.owner[single_mask]
        neighbour = grid.neighbour[single_mask]

        upper_f = self.upper[single_mask]
        lower_f = self.lower[single_mask]
        flux_data = upper_f * psi[neighbour] - lower_f * psi[owner]

        if self._face_flux_correction is not None:
            flux_data = flux_data + self._face_flux_correction

        return flux_data

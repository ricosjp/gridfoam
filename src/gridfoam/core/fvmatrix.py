from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.shapes import (
    broadcast_entity,
    require_shape,
    validate_component_shape,
)


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
        Number of scalar entries in the target field's physical tensor.
    diag : torch.Tensor
        Diagonal coefficients with shape ``[C]``.
    upper : torch.Tensor
        Upper off-diagonal coefficients with shape ``[F]``.
    lower : torch.Tensor
        Lower off-diagonal coefficients with shape ``[F]``.
    source : torch.Tensor
        Right-hand-side source with shape ``[C, *component_shape]``.
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
        component_shape = field.component_shape

        # LDU
        self._diag = torch.zeros((n_cells,), dtype=dtype, device=device)
        self._upper = torch.zeros((n_faces,), dtype=dtype, device=device)
        self._lower = torch.zeros((n_faces,), dtype=dtype, device=device)
        self._source = torch.zeros(
            (n_cells, *component_shape), dtype=dtype, device=device
        )

        # Explicit face-flux correction on single-sided internal faces,
        # stored by discretization operators (e.g. non-orthogonal Laplacian).
        # OpenFOAM ``fvMatrix::faceFluxCorrectionPtr`` equivalent.
        self._face_flux_correction: (
            Float[torch.Tensor, " F_single *component_shape"] | None
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
        """Number of scalar entries in the target field's physical tensor."""
        return self.field.num_components

    @property
    def diag(self) -> Float[torch.Tensor, " C"]:
        """Return diagonal coefficients."""
        return self._diag

    @diag.setter
    def diag(self, value: Float[torch.Tensor, " C"]):
        """Set diagonal coefficients."""
        require_shape(value, (self.grid.num_cells,), "matrix diag")
        self._diag = value

    @property
    def upper(self) -> Float[torch.Tensor, " F"]:
        """Return upper off-diagonal coefficients."""
        return self._upper

    @upper.setter
    def upper(self, value: Float[torch.Tensor, " F"]):
        """Set upper off-diagonal coefficients."""
        require_shape(value, (self.grid.num_internal_faces,), "matrix upper")
        self._upper = value

    @property
    def lower(self) -> Float[torch.Tensor, " F"]:
        """Return lower off-diagonal coefficients."""
        return self._lower

    @lower.setter
    def lower(self, value: Float[torch.Tensor, " F"]):
        """Set lower off-diagonal coefficients."""
        require_shape(value, (self.grid.num_internal_faces,), "matrix lower")
        self._lower = value

    @property
    def source(self) -> Float[torch.Tensor, " C *component_shape"]:
        """Return right-hand-side source term."""
        return self._source

    @source.setter
    def source(self, value: Float[torch.Tensor, " C *component_shape"]):
        """Set right-hand-side source term."""
        require_shape(
            value,
            (self.grid.num_cells, *self.field.component_shape),
            "matrix source",
        )
        self._source = value

    @property
    def face_flux_correction(
        self,
    ) -> Float[torch.Tensor, " F_single *component_shape"] | None:
        """
        Explicit face-flux correction on single-sided internal faces.

        Set by discretization operators (e.g. ``laplacian``) so that
        ``flux(psi)`` includes non-orthogonal contributions. Corresponds to
        OpenFOAM ``faceFluxCorrectionPtr``.
        """
        return self._face_flux_correction

    @face_flux_correction.setter
    def face_flux_correction(
        self, value: Float[torch.Tensor, " F_single *component_shape"] | None
    ):
        """Set the explicit face-flux correction on single-sided faces."""
        if value is not None:
            require_shape(
                value,
                (
                    int(self._single_internal_mask().sum()),
                    *self.field.component_shape,
                ),
                "face flux correction",
            )
        self._face_flux_correction = value

    @staticmethod
    def _combine_face_flux_correction(
        a: Float[torch.Tensor, " F_single *component_shape"] | None,
        b: Float[torch.Tensor, " F_single *component_shape"] | None,
        sign: float,
    ) -> Float[torch.Tensor, " F_single *component_shape"] | None:
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

    def with_source(
        self, source: Float[torch.Tensor, " C *component_shape"]
    ) -> FvMatrix:
        """Return an independent matrix with a replacement integrated RHS.

        Copy all coefficient and face-correction tensors while preserving
        autograd connections. The field and grid remain shared. In momentum
        prediction this adds the pressure force to a separate solve matrix,
        leaving the pressure-free equation available for H/A.

        Parameters
        ----------
        source : torch.Tensor
            Replacement volume-integrated RHS, shape ``[C, *component_shape]``.

        Returns
        -------
        FvMatrix
            Matrix with independent tensor storage and the same field/grid.
        """
        result = FvMatrix(self.field)
        result.source = source.clone()
        result.diag = self.diag.clone()
        result.upper = self.upper.clone()
        result.lower = self.lower.clone()
        if self.face_flux_correction is not None:
            result.face_flux_correction = self.face_flux_correction.clone()
        return result

    def A(self) -> Float[torch.Tensor, " C"]:
        """
        OpenFOAM-style ``A()`` operator.

        Returns volume-normalized diagonal entries: ``diag / V``.
        """
        return self.diag / self.grid.cell_volumes

    def H1(self) -> Float[torch.Tensor, " C"]:
        """
        OpenFOAM-style ``H1()`` operator.

        Returns the negative row sum of the off-diagonal coefficients
        normalised by the cell volume, ``-sum_N a_PN / V``. Used by the
        SIMPLEC (``consistent``) formulation:
        ``rAtU = 1 / (1/rAU - H1)``. Boundary contributions are already
        folded into ``diag`` and therefore do not appear here, matching
        OpenFOAM for non-coupled patches.
        """
        h1 = torch.zeros_like(self.diag)
        h1.index_add_(0, self.grid.owner, -self.upper)
        h1.index_add_(0, self.grid.neighbour, -self.lower)
        return h1 / self.grid.cell_volumes

    def H(
        self, x: Float[torch.Tensor, " C *component_shape"]
    ) -> Float[torch.Tensor, " C *component_shape"]:
        """
        OpenFOAM-style ``H()`` operator.

        Computes and returns:
        ``H = (b - (A - diag) * x) / V = (b - A * x + diag * x) / V``.

        Parameters
        ----------
        x : torch.Tensor
            Current field values with shape ``[C, *component_shape]``.

        Returns
        -------
        torch.Tensor
            Computed H values with shape ``[C, *component_shape]``.
        """
        require_shape(
            x, (self.grid.num_cells, *self.field.component_shape), "H input"
        )
        diag = broadcast_entity(self.diag, x)
        volumes = broadcast_entity(self.grid.cell_volumes, x)
        return (self.source - self.multiply(x) + diag * x) / volumes

    def multiply(
        self, x: Float[torch.Tensor, " C *component_shape"]
    ) -> Float[torch.Tensor, " C *component_shape"]:
        """
        Compute matrix-vector product ``A * x``.

        Parameters
        ----------
        x : torch.Tensor
            Input vector with shape ``[C, *component_shape]``.

        Returns
        -------
        torch.Tensor
            Product vector with shape ``[C, *component_shape]``.
        """
        validate_component_shape(tuple(x.shape[1:]))
        owner = self.grid.owner
        neighbour = self.grid.neighbour

        x_owner = x[owner]
        x_neighbour = x[neighbour]
        diag = broadcast_entity(self.diag, x)
        upper = broadcast_entity(self.upper, x_neighbour)
        lower = broadcast_entity(self.lower, x_owner)

        # Diagonal and off-diagonal contributions to A x.
        res = diag * x
        res.index_add_(0, owner, upper * x_neighbour)
        res.index_add_(0, neighbour, lower * x_owner)

        return res

    def with_fixed_values(
        self, cells: torch.Tensor, values: torch.Tensor
    ) -> FvMatrix:
        """Eliminate prescribed cell values, preserving matrix symmetry.

        Apply this after assembling the complete equation and explicit sources.
        Move prescribed column contributions to the RHS and remove both row
        and column couplings. Each fixed row retains its nonzero diagonal;
        a zero diagonal becomes -1 if the diagonal sum is negative, else +1.
        Scale its RHS by the same diagonal to impose the prescribed value.

        Parameters
        ----------
        cells : torch.Tensor
            Unique cell indices, shape ``[K]``, on the matrix device.
        values : torch.Tensor
            Prescribed values, shape ``[K, *component_shape]``. All physical
            components share the selected cells; gradients flow to values.

        Returns
        -------
        FvMatrix
            Separate solve matrix without face-flux corrections, or ``self``
            if ``cells`` is empty. The input is unchanged; retain it for
            physical face-flux reconstruction.
        """
        require_shape(
            values, (cells.numel(), *self.field.component_shape), "fixed values"
        )
        if cells.numel() == 0:
            return self
        fixed = torch.zeros(
            self.grid.num_cells, dtype=torch.bool, device=cells.device
        )
        fixed[cells] = True
        prescribed = torch.zeros_like(self.source).index_copy(0, cells, values)
        owner, neighbour = self.grid.owner, self.grid.neighbour
        result = FvMatrix(self.field)
        # An isolated boundary cell can have a zero PDE row. Match the
        # sign of the other rows so negative Laplacians stay definite too.
        unit_diag = torch.where(
            self.diag.sum() < 0,
            -torch.ones_like(self.diag),
            torch.ones_like(self.diag),
        )
        constraint_diag = torch.where(self.diag != 0, self.diag, unit_diag)
        result.diag = torch.where(fixed, constraint_diag, self.diag)
        connected = fixed[owner] | fixed[neighbour]
        result.upper = torch.where(
            connected, torch.zeros_like(self.upper), self.upper
        )
        result.lower = torch.where(
            connected, torch.zeros_like(self.lower), self.lower
        )
        source = self.source.index_add(
            0,
            owner,
            -broadcast_entity(self.upper, prescribed[neighbour])
            * prescribed[neighbour],
        )
        source = source.index_add(
            0,
            neighbour,
            -broadcast_entity(self.lower, prescribed[owner])
            * prescribed[owner],
        )
        result.source = source.index_copy(
            0, cells, broadcast_entity(constraint_diag[cells], values) * values
        )
        return result

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
        self, psi: Float[torch.Tensor, " C *component_shape"]
    ) -> Float[torch.Tensor, " F_single *component_shape"]:
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
            Field values with shape ``[C, *component_shape]``.

        Returns
        -------
        torch.Tensor
            Face flux on single-sided internal faces with shape
            ``[F_single, *component_shape]``.
        """
        require_shape(
            psi,
            (self.grid.num_cells, *self.field.component_shape),
            "flux input",
        )
        grid = self.grid
        single_mask = self._single_internal_mask()
        owner = grid.owner[single_mask]
        neighbour = grid.neighbour[single_mask]

        psi_owner = psi[owner]
        psi_neighbour = psi[neighbour]
        upper_f = broadcast_entity(self.upper[single_mask], psi_neighbour)
        lower_f = broadcast_entity(self.lower[single_mask], psi_owner)
        flux_data = upper_f * psi_neighbour - lower_f * psi_owner

        if self._face_flux_correction is not None:
            flux_data = flux_data + self._face_flux_correction

        return flux_data

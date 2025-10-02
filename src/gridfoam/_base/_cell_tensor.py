from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int32

from gridfoam._base._face_tensor import FaceTensor
from gridfoam._base._tvd_scheme import tvd_scheme
from gridfoam._interface._tvd_scheme import ITVDScheme
from gridfoam.utils.enums import TVDScheme


@dataclass
class CellTensor:
    """
    A tensor representing cell-centered data with halo regions.

    This class provides a structured way to handle 3D cell-centered data
    with interior and halo regions. The halo regions are used for
    boundary conditions and communication between adjacent cells.

    Parameters
    ----------
    w_interior : int
        Width of the interior region (excluding halo).
    w_halo : int
        Width of the halo region on each side.
    ndim : int
        Number of dimensions for the field data (0-2).
    raw : torch.Tensor
        Raw tensor data with shape (..., w_interior + 2*w_halo, ...).
    """

    w_interior: int
    w_halo: int
    ndim: int
    raw: torch.Tensor

    @classmethod
    def init(
        cls,
        w_interior: int,
        w_halo: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> CellTensor:
        """
        Initialize a CellTensor with the specified parameters.

        Parameters
        ----------
        w_interior : int
            Width of the interior region.
        w_halo : int
            Width of the halo region on each side.
        shape : tuple[int, ...]
            Shape of the field data (0-2 dimensions).
        dtype : torch.dtype
            Data type of the tensor.
        device : torch.device
            Device where the tensor will be allocated.

        Returns
        -------
        CellTensor
            Initialized CellTensor instance.

        Raises
        ------
        ValueError
            If the shape has invalid dimensions (not 0-2).
        """
        data_width = w_interior + 2 * w_halo
        ndim = len(shape)
        if ndim == 0 or ndim > 2:
            raise ValueError(f"Invalid shape: {shape}")
        shape = (*shape, data_width, data_width, data_width)
        raw = torch.zeros(shape, device=device, dtype=dtype)
        return cls(w_interior, w_halo, ndim, raw)

    @property
    def interior_shape(self) -> tuple[int, ...]:
        """
        Get the shape of the interior region.

        Returns
        -------
        tuple[int, ...]
            Shape of the interior region including field dimensions
            and interior spatial dimensions.
        """
        elem_shape = self.raw.shape[:-3]
        return (*elem_shape, self.w_interior, self.w_interior, self.w_interior)

    @property
    def interior(self) -> torch.Tensor:
        """
        Get the interior region of the tensor.

        Returns
        -------
        torch.Tensor
            View of the interior region (excluding halo).
        """
        return self.raw[
            ...,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo,
        ]

    @interior.setter
    def interior(self, value: torch.Tensor) -> None:
        """
        Set the interior region of the tensor.

        Parameters
        ----------
        value : torch.Tensor
            Values to set in the interior region.
            Must have the same shape as the interior region.
        """
        self.raw[
            ...,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo,
        ] = value

    def get_half_interior(
        self, offset: Int32[torch.Tensor, " 3"]
    ) -> torch.Tensor:
        """
        Get the half interior slice along the specified offset.
        Given the interior width is 8 and
        offset is [1, 0, 0] (+x, -y, -z),
        this method will slice the interior tensor as follows:
        ```python
        tensor[..., 0:4, 0:4, 4:8]
        ```

        Parameters
        ----------
        offset : Int32[torch.Tensor, " 3"]
            Offset to get the half interior slice along.
            Each element should be 0 or 1.
            - 0: slice the left half
            - 1: slice the right half

        Returns
        -------
        torch.Tensor
            Half interior slice along the specified offset.
        """
        w_half = self.w_interior // 2
        slices = [
            slice(
                self.w_halo + offset[i] * w_half,
                -self.w_halo - (1 - offset[i]) * w_half,
            )
            for i in reversed(range(3))
        ]
        slices = [slice(None)] * self.ndim + slices
        return self.raw[tuple(slices)]

    def get_halo_along(self, axis: int, forward: bool) -> torch.Tensor:
        """
        Get the halo slice along the specified axis.

        Parameters
        ----------
        axis : int
            Axis to get the halo slice along.
            - 0: z-axis
            - 1: y-axis
            - 2: x-axis
        forward : bool
            Forward or backward slicing.
            - True: Forward slicing (e.g. +x, +y, +z)
            - False: Backward slicing (e.g. -x, -y, -z)

        Returns
        -------
        torch.Tensor
            Halo slice along the specified axis.
        """
        bnd_slice = (
            slice(-self.w_halo, None) if forward else slice(0, self.w_halo)
        )
        slices = [slice(self.w_halo, -self.w_halo)] * 3
        slices[axis] = bnd_slice
        slices = [slice(None)] * self.ndim + slices
        return self.raw[tuple(slices)]

    def set_halo_along(
        self, axis: int, forward: bool, value: torch.Tensor
    ) -> None:
        """
        Set the halo slice along the specified axis.

        Parameters
        ----------
        axis : int
            Axis to set the halo slice along.
            - 0: z-axis
            - 1: y-axis
            - 2: x-axis
        forward : bool
            Forward or backward slicing.
            - True: Forward slicing (e.g. +x, +y, +z)
            - False: Backward slicing (e.g. -x, -y, -z)
        value : torch.Tensor
            Value to set the halo slice along.
            The shape of the value must be the same as the halo slice.
        """
        bnd_slice = (
            slice(-self.w_halo, None) if forward else slice(0, self.w_halo)
        )
        slices = [slice(self.w_halo, -self.w_halo)] * 3
        slices[axis] = bnd_slice
        slices = [slice(None)] * self.ndim + slices
        self.raw[tuple(slices)] = value

    def get_interior_halo_along(self, axis: int, forward: bool) -> torch.Tensor:
        """
        Get the interior halo slice along the specified axis,
        which is used to update the halo slice.
        Example:
        ```python
        forward = True
        xp_cube_halo_xm = xp_cube.field_tensors[name].get_interior_halo_along(
            2, not forward
        )
        cube.field_tensors[name].set_halo_along(2, forward, xp_cube_halo_xm)
        ```
        where cube is cube_{x,y,z} and xp_cube is cube_{x+1,y,z}

        Parameters
        ----------
        axis : int
            Axis to get the interior slice along.
            - 0: z-axis
            - 1: y-axis
            - 2: x-axis
        forward : bool
            Forward or backward slicing.
            - True: Forward slicing (e.g. +x, +y, +z)
            - False: Backward slicing (e.g. -x, -y, -z)

        Returns
        -------
        torch.Tensor
            Interior halo slice along the specified axis.
        """
        bnd_slice = (
            slice(-2 * self.w_halo, -self.w_halo)
            if forward
            else slice(self.w_halo, 2 * self.w_halo)
        )
        slices = [slice(self.w_halo, -self.w_halo)] * 3
        slices[axis] = bnd_slice
        slices = [slice(None)] * self.ndim + slices
        return self.raw[tuple(slices)]

    def get_shifted_interior_along(
        self, axis: int, forward: bool
    ) -> torch.Tensor:
        """
        Get the shifted interior slice along the specified axis.

        Parameters
        ----------
        axis : int
            Axis to get the shifted interior slice along.
        forward : bool
            Forward or backward slicing.
            - True: Forward slicing (e.g. +x, +y, +z)
            - False: Backward slicing (e.g. -x, -y, -z)

        Returns
        -------
        torch.Tensor
            Shifted interior slice along the specified axis.
        """
        dir = 1 if forward else -1
        slices = [slice(self.w_halo, -self.w_halo)] * 3
        slices[axis] = slice(self.w_halo + dir, -self.w_halo + dir)
        slices = [slice(None)] * self.ndim + slices
        return self.raw[tuple(slices)]

    def face_average(self) -> FaceTensor:
        """
        Compute face-averaged values from cell-centered data.

        This method computes the average of adjacent cell values
        to obtain face-centered values for all six faces.

        Returns
        -------
        FaceTensor
            Face tensor containing averaged values for all faces.
        """

        def axis_face_average(axis: int) -> torch.Tensor:
            """Compute face average for a given axis."""
            # Base interior slice
            base_slice = [slice(None)] * self.ndim + [
                slice(self.w_halo, -self.w_halo)
            ] * 3

            # Forward and backward slices
            forward_slice = base_slice.copy()
            forward_slice[axis] = slice(self.w_halo, -self.w_halo + 1)

            backward_slice = base_slice.copy()
            backward_slice[axis] = slice(self.w_halo - 1, -self.w_halo)

            return 0.5 * (
                self.raw[tuple(forward_slice)] + self.raw[tuple(backward_slice)]
            )

        # Compute face averages for all directions
        xf = axis_face_average(2)  # x-direction
        yf = axis_face_average(1)  # y-direction
        zf = axis_face_average(0)  # z-direction

        return FaceTensor(self.w_interior, xf, yf, zf)

    def face_tensor_for_advection(
        self, U_f: FaceTensor, scheme: TVDScheme
    ) -> FaceTensor:
        """
        Compute face values for advection.
        """
        scheme: ITVDScheme = tvd_scheme(scheme)

        def field_slice(axis: int, shift: int) -> slice:
            """Get field slice along specified axis with shift."""
            start = self.w_halo + shift
            end = -self.w_halo + shift + 1
            if end == 0:
                end = None
            i_slices = [slice(self.w_halo, -self.w_halo)] * 3
            i_slices[axis] = slice(start, end)
            i_slices = [slice(None)] * self.ndim + i_slices
            return self.raw[tuple(i_slices)]

        def compute_face_values(
            axis: int, velocity_face: torch.Tensor
        ) -> torch.Tensor:
            """Compute face values for a given axis using TVD scheme."""
            # Get field values at different positions
            f_m2 = field_slice(axis, -2)  # F_{i-2}
            f_m1 = field_slice(axis, -1)  # F_{i-1}
            f_0 = field_slice(axis, 0)  # F_i
            f_1 = field_slice(axis, 1)  # F_{i+1}

            # Determine upwind and downwind cells
            upcell = velocity_face.sign() >= 0
            downcell = velocity_face.sign() < 0

            # Initialize face values
            face_values = torch.zeros_like(f_0)

            # Upwind scheme with TVD correction
            face_values[..., upcell] = (
                f_m1 + scheme.correction_term(f_m1 - f_m2, f_0 - f_m1)
            )[..., upcell]

            # Downwind scheme with TVD correction
            face_values[..., downcell] = (
                f_0 - scheme.correction_term(f_0 - f_m1, f_1 - f_0)
            )[..., downcell]

            return face_values

        # Initialize face tensor
        face_tensor = FaceTensor.init(
            self.w_interior,
            self.raw.shape[:-3],
            self.raw.dtype,
            self.raw.device,
        )

        # Compute face values for each direction
        face_tensor.x = compute_face_values(2, U_f.x[0])  # x-direction (axis=2)
        face_tensor.y = compute_face_values(1, U_f.y[1])  # y-direction (axis=1)
        face_tensor.z = compute_face_values(0, U_f.z[2])  # z-direction (axis=0)
        return face_tensor


def grad(field: CellTensor, dx: Float[torch.Tensor, " 3"]) -> FaceTensor:
    """
    Compute the gradient of a cell-centered field.

    This function computes the finite difference gradient
    of a cell-centered field using central differences.

    Parameters
    ----------
    field : CellTensor
        Cell-centered field to compute gradient for.
    dx : Float[torch.Tensor, " 3"]
        Grid spacing in each direction [dx, dy, dz].

    Returns
    -------
    FaceTensor
        Face tensor containing the gradient components.
    """

    def axis_diff(axis: int) -> torch.Tensor:
        """Compute face difference for a given axis."""
        # Base interior slice
        base_slice = [slice(None)] * field.ndim + [
            slice(field.w_halo, -field.w_halo)
        ] * 3

        # Forward and backward slices
        forward_slice = base_slice.copy()
        forward_slice[axis] = slice(field.w_halo, -field.w_halo + 1)

        backward_slice = base_slice.copy()
        backward_slice[axis] = slice(field.w_halo - 1, -field.w_halo)

        return (
            field.raw[tuple(forward_slice)] - field.raw[tuple(backward_slice)]
        )

    ddx = axis_diff(2) / dx[0]
    ddy = axis_diff(1) / dx[1]
    ddz = axis_diff(0) / dx[2]
    return FaceTensor(field.w_interior, ddx, ddy, ddz)

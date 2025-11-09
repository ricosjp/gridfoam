from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Float, UInt32

from gridfoam._base._field._tensor import FaceTensor
from gridfoam._base._interface._tvd_scheme import ITVDScheme
from gridfoam.utils.enums import Axis


class CellTensor:
    """
    A tensor representing cell-centered data with halo regions.

    This class provides a structured way to handle 3D cell-centered data
    with interior and halo regions. The halo regions are used for
    boundary conditions and communication between adjacent cells.
    """

    def __init__(
        self,
        N: int,
        H: int,
        C: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """
        Initialize a CellTensor with the specified parameters.

        Parameters
        ----------
        N : int
            Width of the interior region.
        H : int
            Width of the halo region on each side.
        C : int
            Number of channels.
        dtype : torch.dtype
            Data type of the tensor.
        device : torch.device
            Device where the tensor will be allocated.

        Returns
        -------
        CellTensor
            Initialized CellTensor instance.
        """
        self._N = N
        self._H = H
        self._C = C
        W = N + 2 * H
        self._dim = 3
        self._raw = torch.zeros((C, W, W, W), dtype=dtype, device=device)

    @classmethod
    def zeros_like(cls, other: CellTensor) -> CellTensor:
        return cls(other.N, other.H, other.C, other.raw.dtype, other.raw.device)

    def __add__(self, other: CellTensor) -> CellTensor:
        cell_tensor = CellTensor.zeros_like(self)
        cell_tensor.raw = self.raw + other.raw
        return cell_tensor

    def __sub__(self, other: CellTensor) -> CellTensor:
        cell_tensor = CellTensor.zeros_like(self)
        cell_tensor.raw = self.raw - other.raw
        return cell_tensor

    def __mul__(self, other: CellTensor | float) -> CellTensor:
        if isinstance(other, CellTensor):
            cell_tensor = CellTensor.zeros_like(self)
            cell_tensor.raw = self.raw * other.raw
            return cell_tensor
        if isinstance(other, float):
            cell_tensor = CellTensor.zeros_like(self)
            cell_tensor.raw = self.raw * other
            return cell_tensor
        raise ValueError(f"Unsupported operand type for *: '{type(other)}'")

    def __rmul__(self, other: float) -> CellTensor:
        return self * other

    @property
    def N(self) -> int:
        return self._N

    @property
    def H(self) -> int:
        return self._H

    @property
    def C(self) -> int:
        return self._C

    @property
    def raw(self) -> Float[torch.Tensor, "C W W W"]:
        return self._raw

    @raw.setter
    def raw(self, value: Float[torch.Tensor, "C W W W"]) -> None:
        self._raw = value

    @property
    def interior(
        self,
    ) -> Float[torch.Tensor, "C N N N"]:
        """
        Get the interior region of the tensor.

        Returns
        -------
        torch.Tensor
            View of the interior region (excluding halo).
        """
        return self._raw[
            :,
            self._H : -self._H,
            self._H : -self._H,
            self._H : -self._H,
        ]

    @interior.setter
    def interior(self, value: Float[torch.Tensor, "C N N N"]) -> None:
        """
        Set the interior region of the tensor.

        Parameters
        ----------
        value : torch.Tensor
            Values to set in the interior region.
            Must have the same shape as the interior region.
        """
        self._raw[
            :,
            self._H : -self._H,
            self._H : -self._H,
            self._H : -self._H,
        ] = value

    def get_half_interior(
        self, offset: UInt32[np.ndarray, " 3"]
    ) -> Float[torch.Tensor, "C halfN halfN halfN"]:
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
        offset : UInt32[np.ndarray, " 3"]
            Offset to get the half interior slice along.
            Each element should be 0 or 1.
            - 0: slice the left half
            - 1: slice the right half

        Returns
        -------
        torch.Tensor
            Half interior slice along the specified offset.
        """
        offset = offset.astype(int)
        halfN = self._N // 2
        slices = [
            slice(
                self._H + offset[i] * halfN,
                -self._H - (1 - offset[i]) * halfN,
            )
            for i in reversed(range(3))
        ]
        slices = [slice(None)] + slices
        return self._raw[tuple(slices)]

    def get_halo_along(self, axis: Axis, forward: bool) -> torch.Tensor:
        """
        Get the halo slice along the specified axis.

        Parameters
        ----------
        axis : Axis
            Axis to get the halo slice along.
        forward : bool
            Forward or backward slicing.
            - True: Forward slicing (e.g. +x, +y, +z)
            - False: Backward slicing (e.g. -x, -y, -z)

        Returns
        -------
        torch.Tensor
            Halo slice along the specified axis.
        """
        axis = self._dim - axis.value - 1 # convert to 0-based index
        bnd_slice = slice(-self._H, None) if forward else slice(0, self._H)
        slices = [slice(self._H, -self._H)] * 3
        slices[axis] = bnd_slice
        slices = [slice(None)] + slices
        return self._raw[tuple(slices)]

    def set_halo_along(
        self, axis: Axis, forward: bool, value: torch.Tensor
    ) -> None:
        """
        Set the halo slice along the specified axis.

        Parameters
        ----------
        axis : Axis
            Axis to set the halo slice along.
        forward : bool
            Forward or backward slicing.
            - True: Forward slicing (e.g. +x, +y, +z)
            - False: Backward slicing (e.g. -x, -y, -z)
        value : torch.Tensor
            Value to set the halo slice along.
            The shape of the value must be the same as the halo slice.
        """
        axis = self._dim - axis.value - 1 # convert to 0-based index
        bnd_slice = slice(-self._H, None) if forward else slice(0, self._H)
        slices = [slice(self._H, -self._H)] * 3
        slices[axis] = bnd_slice
        slices = [slice(None)] + slices
        self._raw[tuple(slices)] = value

    def get_interior_halo_along(self, axis: Axis, forward: bool) -> torch.Tensor:
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
        axis : Axis
            Axis to get the interior slice along.
        forward : bool
            Forward or backward slicing.
            - True: Forward slicing (e.g. +x, +y, +z)
            - False: Backward slicing (e.g. -x, -y, -z)

        Returns
        -------
        torch.Tensor
            Interior halo slice along the specified axis.
        """
        axis = self._dim - axis.value - 1 # convert to 0-based index
        bnd_slice = (
            slice(-2 * self._H, -self._H)
            if forward
            else slice(self._H, 2 * self._H)
        )
        slices = [slice(self._H, -self._H)] * 3
        slices[axis] = bnd_slice
        slices = [slice(None)] + slices
        return self._raw[tuple(slices)]

    def get_shifted_interior_along(
        self, axis: Axis, shift: int
    ) -> Float[torch.Tensor, "... w w w"]:
        """
        Get the shifted interior slice along the specified axis.

        Parameters
        ----------
        axis : Axis
            Axis to get the shifted interior slice along.
        shift : int
            Shift amount along the specified axis.
            - Positive: Shift forward (e.g. +x, +y, +z)
            - Negative: Shift backward (e.g. -x, -y, -z)

        Returns
        -------
        torch.Tensor
            Shifted interior slice along the specified axis.
        """
        axis = self._dim - axis.value - 1 # convert to 0-based index
        start = self._H + shift
        end = -self._H + shift
        if end == 0:
            end = None
        slices = [slice(self._H, -self._H)] * 3
        slices[axis] = slice(start, end)
        slices = [slice(None)] + slices
        return self._raw[tuple(slices)]

    def get_shifted_interior_along_for_face(
        self, axis: Axis, shift: int
    ) -> torch.Tensor:
        """
        Get the shifted interior slice along the specified axis.
        This method is used
        to get the face-centered values from the cell-centered values.

        Parameters
        ----------
        axis : Axis
            Axis to get the shifted interior slice along.
        shift : int
            Shift amount along the specified axis.
            - Positive: Shift forward (e.g. +x, +y, +z)
            - Negative: Shift backward (e.g. -x, -y, -z)

        Returns
        -------
        torch.Tensor
            Shifted interior slice along the specified axis.
        """
        axis = self._dim - axis.value - 1 # convert to 0-based index
        start = self._H + shift
        end = -self._H + shift + 1
        if end == 0:
            end = None
        slices = [slice(self._H, -self._H)] * 3
        slices[axis] = slice(start, end)
        slices = [slice(None)] + slices
        return self._raw[tuple(slices)]

    def face_average_along(self, axis: Axis) -> torch.Tensor:
        """
        Get the face-averaged values along the specified axis.

        Parameters
        ----------
        axis : Axis
            Axis to compute the face-averaged values along.
        """
        forward = self.get_shifted_interior_along_for_face(axis, 0)
        backward = self.get_shifted_interior_along_for_face(axis, -1)
        return 0.5 * (forward + backward)

    def face_diff_along(self, axis: Axis) -> torch.Tensor:
        """
        Get the face-differenced values along the specified axis.

        Parameters
        ----------
        axis : Axis
            Axis to compute the face-differenced values along.
        """
        forward = self.get_shifted_interior_along_for_face(axis, 0)
        backward = self.get_shifted_interior_along_for_face(axis, -1)
        return forward - backward

    def face_tensor_for_advection(
        self, U_f: FaceTensor, scheme: ITVDScheme
    ) -> FaceTensor:
        def _compute_face_values(
            axis: Axis, velocity_face: torch.Tensor
        ) -> torch.Tensor:
            """Compute face values for a given axis using TVD scheme."""
            # Get field values at different positions
            f_m2 = self.get_shifted_interior_along_for_face(axis, -2)  # F_{i-2}
            f_m1 = self.get_shifted_interior_along_for_face(axis, -1)  # F_{i-1}
            f_0 = self.get_shifted_interior_along_for_face(axis, 0)  # F_i
            f_1 = self.get_shifted_interior_along_for_face(axis, 1)  # F_{i+1}

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
        face_tensor = FaceTensor(self.N, self.C, U_f.x.dtype, U_f.x.device)
        face_tensor.x = _compute_face_values(Axis.X, U_f.x[Axis.X.value])
        face_tensor.y = _compute_face_values(Axis.Y, U_f.y[Axis.Y.value])
        face_tensor.z = _compute_face_values(Axis.Z, U_f.z[Axis.Z.value])
        return face_tensor

    def face_average(self) -> FaceTensor:
        """
        Get the face-averaged values.

        Returns
        -------
        FaceTensor
            Face-averaged tensor.
        """
        face_tensor = FaceTensor(self.N, self.C, self.raw.dtype, self.raw.device)
        face_tensor.x = self.face_average_along(Axis.X)
        face_tensor.y = self.face_average_along(Axis.Y)
        face_tensor.z = self.face_average_along(Axis.Z)
        return face_tensor

    def grad(self, dx: Float[torch.Tensor, " 3"]) -> FaceTensor:
        """
        Compute the gradient of the cell-centered tensor.

        Parameters
        ----------
        dx : Float[torch.Tensor, " 3"]
            Grid spacing in each direction.

        Returns
        -------
        FaceTensor
            Gradient tensor.
        """
        grad_tensor = FaceTensor(
            self.N, self.C, self.raw.dtype, self.raw.device
        )
        grad_tensor.x = self.face_diff_along(Axis.X) / dx[Axis.X.value]
        grad_tensor.y = self.face_diff_along(Axis.Y) / dx[Axis.Y.value]
        grad_tensor.z = self.face_diff_along(Axis.Z) / dx[Axis.Z.value]
        return grad_tensor

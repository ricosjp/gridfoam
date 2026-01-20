from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Float, UInt32

from gridfoam.DNA.enum import Axis
from gridfoam.DNA.fielddata._face import FaceField
from gridfoam.DNA.utils.harmonic_mean import harmonic_mean


class CellField:
    """
    A field representing cell-centered data with halo regions.

    This class provides a structured way to handle 3D cell-centered data
    with interior and halo regions. The halo regions are used for
    boundary conditions and communication between adjacent cells.
    """

    def __init__(
        self,
        T: int,
        C: int,
        N: int,
        H: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """
        Initialize a CellField with the specified parameters.

        Parameters
        ----------
        T : int
            Number of time levels.
        C : int
            The number of components of the tensor.
        N : int
            Width of the interior region.
        H : int
            Width of the halo region on each side.
        dtype : torch.dtype
            Data type of the tensor.
        device : torch.device
            Device where the tensor will be allocated.

        Returns
        -------
        CellTensor
            Initialized CellTensor instance.
        """
        self._T = T
        self._C = C
        self._N = N
        self._H = H
        W = N + 2 * H
        self._dim = 3
        self._raw = torch.zeros((T, C, W, W, W), dtype=dtype, device=device)

    @property
    def T(self) -> int:
        return self._T

    @property
    def C(self) -> int:
        return self._C

    @property
    def N(self) -> int:
        return self._N

    @property
    def H(self) -> int:
        return self._H

    @property
    def raw(self) -> Float[torch.Tensor, "T C W W W"]:
        return self._raw

    @raw.setter
    def raw(self, value: Float[torch.Tensor, "T C W W W"]) -> None:
        self._raw = value

    @property
    def interior(
        self,
    ) -> Float[torch.Tensor, "T C N N N"]:
        """
        Get the interior region of the tensor.

        Returns
        -------
        torch.Tensor
            View of the interior region (excluding halo).
        """
        return self._raw[
            :,
            :,
            self._H : -self._H,
            self._H : -self._H,
            self._H : -self._H,
        ]

    @interior.setter
    def interior(self, value: Float[torch.Tensor, "T C N N N"]) -> None:
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
            :,
            self._H : -self._H,
            self._H : -self._H,
            self._H : -self._H,
        ] = value

    @classmethod
    def zeros_like(cls, other: CellField) -> CellField:
        return cls(
            other.T,
            other.C,
            other.N,
            other.H,
            other.raw.dtype,
            other.raw.device,
        )

    def __iadd__(self, other: CellField) -> CellField:
        self.raw += other.raw
        return self

    def __isub__(self, other: CellField) -> CellField:
        self.raw -= other.raw
        return self

    def __imul__(self, other: CellField | float) -> CellField:
        if isinstance(other, CellField):
            self.raw *= other.raw
            return self
        if isinstance(other, float):
            self.raw *= other
            return self
        raise ValueError(f"Unsupported operand type for *: '{type(other)}'")

    def __add__(self, other: CellField) -> CellField:
        cell_field = CellField.zeros_like(self)
        cell_field.raw = self.raw + other.raw
        return cell_field

    def __sub__(self, other: CellField) -> CellField:
        cell_field = CellField.zeros_like(self)
        cell_field.raw = self.raw - other.raw
        return cell_field

    def __mul__(self, other: CellField | float) -> CellField:
        if isinstance(other, CellField):
            cell_field = CellField.zeros_like(self)
            cell_field.raw = self.raw * other.raw
            return cell_field
        if isinstance(other, float):
            cell_field = CellField.zeros_like(self)
            cell_field.raw = self.raw * other
            return cell_field
        raise ValueError(f"Unsupported operand type for *: '{type(other)}'")

    def __rmul__(self, other: float) -> CellField:
        return self * other

    def get_boundary_cell_along(
        self, axis: Axis, forward: bool
    ) -> Float[torch.Tensor, "T C N N"]:
        """
        Get the boundary cell value along the specified axis.
        """
        cell = -1 if forward else 0
        match axis:
            case Axis.X:
                return self.interior[:, :, :, :, cell]
            case Axis.Y:
                return self.interior[:, :, :, cell, :]
            case Axis.Z:
                return self.interior[:, :, cell, :, :]
            case _:
                raise ValueError(f"Invalid axis: {axis}")

    def set_boundary_cell_along(
        self, axis: Axis, forward: bool, value: Float[torch.Tensor, "T C N N"]
    ) -> None:
        """
        Set the boundary cell value along the specified axis.
        """
        cell = -1 if forward else 0
        match axis:
            case Axis.X:
                self.interior[:, :, :, :, cell] = value
            case Axis.Y:
                self.interior[:, :, :, cell, :] = value
            case Axis.Z:
                self.interior[:, :, cell, :, :] = value
            case _:
                raise ValueError(f"Invalid axis: {axis}")

    def get_half_interior(
        self, offset: UInt32[np.ndarray, " 3"]
    ) -> Float[torch.Tensor, "T C halfN halfN halfN"]:
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
        Float[torch.Tensor, "T C halfN halfN halfN"]
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
        slices = [slice(None)] * 2 + slices
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
        Float[torch.Tensor, "T C N|H N|H N|H"]
            Halo slice along the specified axis.
            The length along the axis specified by `axis` is H.
        """
        axis = self._dim - axis.value  # convert to 0-based index
        bnd_slice = slice(-self._H, None) if forward else slice(0, self._H)
        slices = [slice(self._H, -self._H)] * 3
        slices[axis] = bnd_slice
        slices = [slice(None)] * 2 + slices
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
        axis = self._dim - axis.value  # convert to 0-based index
        bnd_slice = slice(-self._H, None) if forward else slice(0, self._H)
        slices = [slice(self._H, -self._H)] * 3
        slices[axis] = bnd_slice
        slices = [slice(None)] * 2 + slices
        self._raw[tuple(slices)] = value

    def get_interior_halo_along(
        self, axis: Axis, forward: bool, flip: bool = False
    ) -> torch.Tensor:
        """
        Get the interior halo slice along the specified axis,
        which is used to update the halo slice.

        Parameters
        ----------
        axis : Axis
            Axis to get the interior slice along.
        forward : bool
            Forward or backward slicing.
            - True: Forward slicing (e.g. +x, +y, +z)
            - False: Backward slicing (e.g. -x, -y, -z)
        flip : bool
            Flip the halo slice along the specified axis.
            - True: Flip the halo slice
            - False: Do not flip the halo slice

        Returns
        -------
        Float[torch.Tensor, "T C N|H N|H N|H"]
            Interior halo slice along the specified axis.
            The length along the axis specified by `axis` is H.
        """
        axis = self._dim - axis.value  # convert to 0-based index
        bnd_slice = (
            slice(-2 * self._H, -self._H)
            if forward
            else slice(self._H, 2 * self._H)
        )
        slices = [slice(self._H, -self._H)] * 3
        slices[axis] = bnd_slice
        slices = [slice(None)] * 2 + slices
        ret = self._raw[tuple(slices)]
        if flip:
            return torch.flip(ret, dims=[2 + axis])
        else:
            return ret

    def get_shifted_interior_along(
        self, axis: Axis, shift: int
    ) -> Float[torch.Tensor, "T C N N N"]:
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
        Float[torch.Tensor, "T C N N N"]
            Shifted interior slice along the specified axis.
        """
        axis = self._dim - axis.value  # convert to 0-based index
        start = self._H + shift
        end = -self._H + shift
        if end == 0:
            end = None
        slices = [slice(self._H, -self._H)] * 3
        slices[axis] = slice(start, end)
        slices = [slice(None)] * 2 + slices
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
        Float[torch.Tensor, "T C N|L N|L N|L"]
            Shifted interior slice along the specified axis.
            The length along the axis specified by `axis` is L = N+1.
        """
        axis = self._dim - axis.value  # convert to 0-based index
        start = self._H + shift
        end = -self._H + shift + 1
        if end == 0:
            end = None
        slices = [slice(self._H, -self._H)] * 3
        slices[axis] = slice(start, end)
        slices = [slice(None)] * 2 + slices
        return self._raw[tuple(slices)]

    def face_average_along(self, axis: Axis) -> torch.Tensor:
        """
        Get the face-averaged values along the specified axis.

        Parameters
        ----------
        axis : Axis
            Axis to compute the face-averaged values along.

        Returns
        -------
        Float[torch.Tensor, "T C N|L N|L N|L"]
            Face-averaged values along the specified axis.
            The length along the axis specified by `axis` is L = N+1.
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

        Returns
        -------
        Float[torch.Tensor, "T C N|L N|L N|L"]
            Face-differenced values along the specified axis.
            The length along the axis specified by `axis` is L = N+1.
        """
        forward = self.get_shifted_interior_along_for_face(axis, 0)
        backward = self.get_shifted_interior_along_for_face(axis, -1)
        return forward - backward

    def face_harmonic_mean_along(self, axis: Axis) -> torch.Tensor:
        """
        Get the face-harmonic mean values along the specified axis.

        Parameters
        ----------
        axis : Axis
            Axis to compute the face-harmonic mean values along.
        """
        forward = self.get_shifted_interior_along_for_face(axis, 0)
        backward = self.get_shifted_interior_along_for_face(axis, -1)
        return harmonic_mean(forward, backward)

    def face_average(self) -> FaceField:
        """
        Get the face-averaged values.

        Returns
        -------
        FaceField
            Face-averaged tensor.
        """
        face_field = FaceField(
            self.T, self.C, self.N, self.raw.dtype, self.raw.device
        )
        face_field.x = self.face_average_along(Axis.X)
        face_field.y = self.face_average_along(Axis.Y)
        face_field.z = self.face_average_along(Axis.Z)
        return face_field

    def face_harmonic_mean(self) -> FaceField:
        """
        Get the face-harmonic mean values.

        Returns
        -------
        FaceField
            Face-harmonic mean tensor.
        """
        face_field = FaceField(
            self.T, self.C, self.N, self.raw.dtype, self.raw.device
        )
        face_field.x = self.face_harmonic_mean_along(Axis.X)
        face_field.y = self.face_harmonic_mean_along(Axis.Y)
        face_field.z = self.face_harmonic_mean_along(Axis.Z)
        return face_field

    def face_grad(self, dx: Float[torch.Tensor, " 3"]) -> FaceField:
        """
        Compute the face-centered gradient field.

        Parameters
        ----------
        dx : Float[torch.Tensor, " 3"]
            Grid spacing in each direction.

        Returns
        -------
        FaceField
            Gradient tensor.
        """
        grad_field = FaceField(
            self.T, self.C, self.N, self.raw.dtype, self.raw.device
        )
        grad_field.x = self.face_diff_along(Axis.X) / dx[Axis.X.value - 1]
        grad_field.y = self.face_diff_along(Axis.Y) / dx[Axis.Y.value - 1]
        grad_field.z = self.face_diff_along(Axis.Z) / dx[Axis.Z.value - 1]
        return grad_field

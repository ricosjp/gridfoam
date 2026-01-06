from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.DNA.enum import Axis


class FaceField:
    """
    A field representing face-centered data for finite volume methods.

    This class provides a structured way to handle face-centered data
    for finite volume computations. It stores flux data on the six faces
    of a 3D cell in three separate tensors (x, y, z faces).
    """

    def __init__(
        self,
        T: int,
        C: int,
        N: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """
        Initialize a FaceField with the specified parameters.

        Parameters
        ----------
        T : int
            Number of time levels.
        C : int
            Number of components.
        N : int
            Width of the interior region.
        dtype : torch.dtype
            Data type of the fields.
        device : torch.device
            Device where the fields will be allocated.

        Returns
        -------
        FaceField
            Initialized FaceField instance with zero-filled tensors.
        """
        self._T = T
        self._C = C
        self._N = N
        shape_x = (T, C, N, N, N + 1)
        shape_y = (T, C, N, N + 1, N)
        shape_z = (T, C, N + 1, N, N)
        self._x = torch.zeros(shape_x, device=device, dtype=dtype)
        self._y = torch.zeros(shape_y, device=device, dtype=dtype)
        self._z = torch.zeros(shape_z, device=device, dtype=dtype)

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
    def x(
        self,
    ) -> Float[torch.Tensor, "T C N N L"]:
        return self._x

    @x.setter
    def x(
        self,
        value: Float[torch.Tensor, "T C N N L"],
    ) -> None:
        self._x = value

    @property
    def y(
        self,
    ) -> Float[torch.Tensor, "T C N L N"]:
        return self._y

    @y.setter
    def y(
        self,
        value: Float[torch.Tensor, "T C N L N"],
    ) -> None:
        self._y = value

    @property
    def z(
        self,
    ) -> Float[torch.Tensor, "T C L N N"]:
        return self._z

    @z.setter
    def z(
        self,
        value: Float[torch.Tensor, "T C L N N"],
    ) -> None:
        self._z = value

    @classmethod
    def zeros_like(cls, other: FaceField) -> FaceField:
        return cls(other.T, other.C, other.N, other.x.dtype, other.x.device)

    def __add__(self, other: FaceField) -> FaceField:
        face_field = FaceField.zeros_like(self)
        face_field.x = self.x + other.x
        face_field.y = self.y + other.y
        face_field.z = self.z + other.z
        return face_field

    def __sub__(self, other: FaceField) -> FaceField:
        face_field = FaceField.zeros_like(self)
        face_field.x = self.x - other.x
        face_field.y = self.y - other.y
        face_field.z = self.z - other.z
        return face_field

    def __mul__(self, other: FaceField) -> FaceField:
        face_field = FaceField.zeros_like(self)
        face_field.x = self.x * other.x
        face_field.y = self.y * other.y
        face_field.z = self.z * other.z
        return face_field

    def get_boundary_face_along(
        self, axis: Axis, forward: bool
    ) -> Float[torch.Tensor, "T C N N"]:
        """
        Get the boundary face value along the specified axis.
        Parameters
        ----------
        axis : Axis
            Axis to get the boundary face value along.
        forward : bool
            Forward or backward slicing.
            - True: Forward slicing (e.g. +x, +y, +z)
            - False: Backward slicing (e.g. -x, -y, -z)

        Returns
        -------
        torch.Tensor
            Boundary face value along the specified axis.
        """
        face = self._N if forward else 0
        match axis:
            case Axis.X:
                return self.x[:, :, :, :, face]
            case Axis.Y:
                return self.y[:, :, :, face, :]
            case Axis.Z:
                return self.z[:, :, face, :, :]
            case _:
                raise ValueError(f"Invalid axis: {axis}")

    def set_boundary_face_along(
        self, axis: Axis, forward: bool, value: Float[torch.Tensor, "T C N N"]
    ) -> None:
        """
        Set the boundary face value along the specified axis.
        Parameters
        ----------
        axis : Axis
            Axis to set the boundary face value along.
        forward : bool
            Forward or backward slicing.
            - True: Forward slicing (e.g. +x, +y, +z)
            - False: Backward slicing (e.g. -x, -y, -z)
        value : Float[torch.Tensor, "T C N N"]
            Value to set the boundary face value along.
            The shape of the value must be the same as the boundary face.
        """
        face = self._N if forward else 0
        match axis:
            case Axis.X:
                self.x[:, :, :, :, face] = value
            case Axis.Y:
                self.y[:, :, :, face, :] = value
            case Axis.Z:
                self.z[:, :, face, :, :] = value
            case _:
                raise ValueError(f"Invalid axis: {axis}")

    def get_face_values_around_cell(
        self, axis: Axis, forward: bool
    ) -> Float[torch.Tensor, "T C N N N"]:
        """
        Get the face value along the specified axis.
        """
        face = slice(1, None) if forward else slice(0, -1)
        match axis:
            case Axis.X:
                return self.x[:, :, :, :, face]
            case Axis.Y:
                return self.y[:, :, :, face, :]
            case Axis.Z:
                return self.z[:, :, face, :, :]

    def integrate_dSn(
        self,
        Sf: Float[torch.Tensor, " 3"],
    ) -> Float[torch.Tensor, "T C N N N"]:
        """
        Integrate face fluxes to compute cell-centered divergence.

        This method computes the divergence of the face-centered flux
        by taking the difference between forward and backward faces
        for each direction and summing them up.

        Returns
        -------
        Float[torch.Tensor, "T C N N N"]
            Cell-centered divergence tensor with shape
            (T, C, N, N, N).
        """
        xp = self.x[:, :, :, :, 1:] * Sf[0]
        xm = self.x[:, :, :, :, :-1] * Sf[0]
        yp = self.y[:, :, :, 1:, :] * Sf[1]
        ym = self.y[:, :, :, :-1, :] * Sf[1]
        zp = self.z[:, :, 1:, :, :] * Sf[2]
        zm = self.z[:, :, :-1, :, :] * Sf[2]
        return xp - xm + yp - ym + zp - zm

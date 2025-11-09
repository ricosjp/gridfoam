from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.utils.enums import Axis


class FaceTensor:
    """
    A tensor representing face-centered data for finite volume methods.

    This class provides a structured way to handle face-centered data
    for finite volume computations. It stores flux data on the six faces
    of a 3D cell in three separate tensors (x, y, z faces).
    """

    def __init__(
        self,
        N: int,
        C: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """
        Initialize a FaceTensor with the specified parameters.

        Parameters
        ----------
        N : int
            Width of the interior region.
        C : int
            Number of channels.
        dtype : torch.dtype
            Data type of the tensors.
        device : torch.device
            Device where the tensors will be allocated.

        Returns
        -------
        FaceTensor
            Initialized FaceTensor instance with zero-filled tensors.
        """
        self._N = N
        self._C = C
        shape_x = (C, N, N, N + 1)
        shape_y = (C, N, N + 1, N)
        shape_z = (C, N + 1, N, N)
        self._x = torch.zeros(shape_x, device=device, dtype=dtype)
        self._y = torch.zeros(shape_y, device=device, dtype=dtype)
        self._z = torch.zeros(shape_z, device=device, dtype=dtype)

    @property
    def N(self) -> int:
        return self._N

    @property
    def C(self) -> int:
        return self._C

    @property
    def x(
        self,
    ) -> Float[torch.Tensor, "C N N L"]:
        return self._x

    @x.setter
    def x(
        self,
        value: Float[torch.Tensor, "C N N L"],
    ) -> None:
        self._x = value

    @property
    def y(
        self,
    ) -> Float[torch.Tensor, "C N L N"]:
        return self._y

    @y.setter
    def y(
        self,
        value: Float[torch.Tensor, "C N L N"],
    ) -> None:
        self._y = value

    @property
    def z(
        self,
    ) -> Float[torch.Tensor, "C L N N"]:
        return self._z

    @z.setter
    def z(
        self,
        value: Float[torch.Tensor, "C L N N"],
    ) -> None:
        self._z = value

    @classmethod
    def zeros_like(cls, other: FaceTensor) -> FaceTensor:
        return cls(other.N, other.C, other.x.dtype, other.x.device)

    def __add__(self, other: FaceTensor) -> FaceTensor:
        face_tensor = FaceTensor.zeros_like(self)
        face_tensor.x = self.x + other.x
        face_tensor.y = self.y + other.y
        face_tensor.z = self.z + other.z
        return face_tensor

    def __sub__(self, other: FaceTensor) -> FaceTensor:
        face_tensor = FaceTensor.zeros_like(self)
        face_tensor.x = self.x - other.x
        face_tensor.y = self.y - other.y
        face_tensor.z = self.z - other.z
        return face_tensor

    def __mul__(self, other: FaceTensor) -> FaceTensor:
        face_tensor = FaceTensor.zeros_like(self)
        face_tensor.x = self.x * other.x
        face_tensor.y = self.y * other.y
        face_tensor.z = self.z * other.z
        return face_tensor

    def get_boundary_face_along(
        self, axis: Axis, forward: bool
    ) -> Float[torch.Tensor, "C N N"]:
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
                return self.z[:, face, :, :]
            case Axis.Y:
                return self.y[:, face, :]
            case Axis.Z:
                return self.x[:, face]
            case _:
                raise ValueError(f"Invalid axis: {axis}")

    def calculate_cell_average(
        self,
    ) -> Float[torch.Tensor, "C 3 N N N"]:
        """
        Get cell-centered tensor by averaging the face tensor.

        This property computes cell-centered values by averaging
        the adjacent face values for each direction.

        Returns
        -------
        Float[torch.Tensor, "C 3 N N N"]
            Cell-centered tensor
            with shape (C, 3, N, N, N).
        """
        x = 0.5 * (self.x[:, :, :, 1:] + self.x[:, :, :, :-1])
        y = 0.5 * (self.y[:, :, 1:, :] + self.y[:, :, :-1, :])
        z = 0.5 * (self.z[:, 1:, :, :] + self.z[:, :-1, :, :])
        return torch.stack([x, y, z], dim=-4)

    def integrate_cell(
        self,
    ) -> Float[torch.Tensor, "C N N N"]:
        """
        Integrate face fluxes to compute cell-centered divergence.

        This method computes the divergence of the face-centered flux
        by taking the difference between forward and backward faces
        for each direction and summing them up.

        Returns
        -------
        Float[torch.Tensor, "C N N N"]
            Cell-centered divergence tensor with shape
            (C, N, N, N).
        """
        xp = self.x[:, :, :, 1:]
        xm = self.x[:, :, :, :-1]
        yp = self.y[:, :, 1:, :]
        ym = self.y[:, :, :-1, :]
        zp = self.z[:, 1:, :, :]
        zm = self.z[:, :-1, :, :]
        return xp - xm + yp - ym + zp - zm

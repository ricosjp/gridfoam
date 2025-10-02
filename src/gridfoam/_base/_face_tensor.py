from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float


@dataclass
class FaceTensor:
    """
    A tensor representing face-centered data for finite volume methods.

    This class provides a structured way to handle face-centered data
    for finite volume computations. It stores flux data on the six faces
    of a 3D cell in three separate tensors (x, y, z faces).

    Parameters
    ----------
    w_interior : int
        Width of the interior region.
    x : Float[torch.Tensor, "... w_interior w_interior (w_interior+1)"]
        Face data for x-faces (normal to x-axis).
    y : Float[torch.Tensor, "... w_interior (w_interior+1) w_interior"]
        Face data for y-faces (normal to y-axis).
    z : Float[torch.Tensor, "... (w_interior+1) w_interior w_interior"]
        Face data for z-faces (normal to z-axis).
    """

    w_interior: int
    x: Float[torch.Tensor, "... w_interior w_interior (w_interior+1)"]
    y: Float[torch.Tensor, "... w_interior (w_interior+1) w_interior"]
    z: Float[torch.Tensor, "... (w_interior+1) w_interior w_interior"]

    @classmethod
    def init(
        cls,
        w_interior: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> FaceTensor:
        """
        Initialize a FaceTensor with the specified parameters.

        Parameters
        ----------
        w_interior : int
            Width of the interior region.
        shape : tuple[int, ...]
            Shape of the field data (0-2 dimensions).
        dtype : torch.dtype
            Data type of the tensors.
        device : torch.device
            Device where the tensors will be allocated.

        Returns
        -------
        FaceTensor
            Initialized FaceTensor instance with zero-filled tensors.
        """
        shape_x = (*shape, w_interior, w_interior, w_interior + 1)
        shape_y = (*shape, w_interior, w_interior + 1, w_interior)
        shape_z = (*shape, w_interior + 1, w_interior, w_interior)
        x = torch.zeros(shape_x, device=device, dtype=dtype)
        y = torch.zeros(shape_y, device=device, dtype=dtype)
        z = torch.zeros(shape_z, device=device, dtype=dtype)
        return cls(w_interior, x, y, z)

    def __mul__(self, other: FaceTensor) -> FaceTensor:
        """
        Multiply two FaceTensor instances element-wise.

        Parameters
        ----------
        other : FaceTensor
            Another FaceTensor to multiply with.

        Returns
        -------
        FaceTensor
            New FaceTensor with element-wise multiplication results.
        """
        return FaceTensor(
            self.w_interior,
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
        )

    def get_boundary_face_along(
        self, axis: int, forward: bool
    ) -> Float[torch.Tensor, "... w_interior w_interior"]:
        """
        Get the boundary face value along the specified axis.
        Parameters
        ----------
        axis : int
            Axis to get the boundary face value along.
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
            Boundary face value along the specified axis.
        """
        face = self.w_interior if forward else 0
        match axis:
            case 0:
                return self.z[..., face, :, :]
            case 1:
                return self.y[..., face, :]
            case 2:
                return self.x[..., face]
            case _:
                raise ValueError(f"Invalid axis: {axis}")

    @property
    def interior(
        self,
    ) -> Float[torch.Tensor, "... 3 w_interior w_interior w_interior"]:
        """
        Get cell-centered tensor by averaging the face tensor.

        This property computes cell-centered values by averaging
        the adjacent face values for each direction.

        Returns
        -------
        Float[torch.Tensor, "... 3 w_interior w_interior w_interior"]
            Cell-centered tensor with shape (..., 3, w_interior, w_interior, w_interior).
            The second-to-last dimension contains [x, y, z] components.
        """
        x = 0.5 * (self.x[..., 1:] + self.x[..., :-1])
        y = 0.5 * (self.y[..., 1:, :] + self.y[..., :-1, :])
        z = 0.5 * (self.z[..., 1:, :, :] + self.z[..., :-1, :, :])
        return torch.stack([x, y, z], dim=-4)

    def integrate_cell(
        self,
    ) -> Float[torch.Tensor, "... w_interior w_interior w_interior"]:
        """
        Integrate face fluxes to compute cell-centered divergence.

        This method computes the divergence of the face-centered flux
        by taking the difference between forward and backward faces
        for each direction and summing them up.

        Returns
        -------
        Float[torch.Tensor, "... w_interior w_interior w_interior"]
            Cell-centered divergence tensor with shape
            (..., w_interior, w_interior, w_interior).
        """
        xp = self.x[..., 1:]
        xm = self.x[..., :-1]
        yp = self.y[..., 1:, :]
        ym = self.y[..., :-1, :]
        zp = self.z[..., 1:, :, :]
        zm = self.z[..., :-1, :, :]
        return xp - xm + yp - ym + zp - zm

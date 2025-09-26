from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float


@dataclass
class FaceTensor:
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
        shape_x = (*shape, w_interior, w_interior, w_interior + 1)
        shape_y = (*shape, w_interior, w_interior + 1, w_interior)
        shape_z = (*shape, w_interior + 1, w_interior, w_interior)
        x = torch.zeros(shape_x, device=device, dtype=dtype)
        y = torch.zeros(shape_y, device=device, dtype=dtype)
        z = torch.zeros(shape_z, device=device, dtype=dtype)
        return cls(w_interior, x, y, z)

    def __mul__(self, other: FaceTensor) -> FaceTensor:
        return FaceTensor(
            self.w_interior,
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
        )

    def get_face_along(
        self, axis: int, forward: bool
    ) -> Float[torch.Tensor, "... w_interior w_interior"]:
        """
        Get the face flux along the specified axis.
        Parameters
        ----------
        axis : int
            Axis to get the face flux along.
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
            Face flux along the specified axis.
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
        """
        x = 0.5 * (self.x[..., 1:] + self.x[..., :-1])
        y = 0.5 * (self.y[..., 1:, :] + self.y[..., :-1, :])
        z = 0.5 * (self.z[..., 1:, :, :] + self.z[..., :-1, :, :])
        return torch.stack([x, y, z], dim=-4)

    def integrate_cell(
        self,
    ) -> Float[torch.Tensor, "... w_interior w_interior w_interior"]:
        xp = self.x[..., 1:]
        xm = self.x[..., :-1]
        yp = self.y[..., 1:, :]
        ym = self.y[..., :-1, :]
        zp = self.z[..., 1:, :, :]
        zm = self.z[..., :-1, :, :]
        return xp - xm + yp - ym + zp - zm

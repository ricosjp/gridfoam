from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int32

from gridfoam._base._face_tensor import FaceTensor


@dataclass
class CellTensor:
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
        data_width = w_interior + 2 * w_halo
        ndim = len(shape)
        if ndim == 0 or ndim > 2:
            raise ValueError(f"Invalid shape: {shape}")
        shape = (*shape, data_width, data_width, data_width)
        raw = torch.zeros(shape, device=device, dtype=dtype)
        return cls(w_interior, w_halo, ndim, raw)


    @property
    def interior_shape(self) -> tuple[int, ...]:
        elem_shape = self.raw.shape[:-3]
        return (*elem_shape, self.w_interior, self.w_interior, self.w_interior)

    @property
    def interior(self) -> torch.Tensor:
        return self.raw[
            ...,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo,
        ]

    @interior.setter
    def interior(self, value: torch.Tensor) -> None:
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
        xp = self.raw[
            ...,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo + 1,
        ]
        xm = self.raw[
            ...,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo,
            self.w_halo - 1 : -self.w_halo,
        ]
        yp = self.raw[
            ...,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo + 1,
            self.w_halo : -self.w_halo,
        ]
        ym = self.raw[
            ...,
            self.w_halo : -self.w_halo,
            self.w_halo - 1 : -self.w_halo,
            self.w_halo : -self.w_halo,
        ]
        zp = self.raw[
            ...,
            self.w_halo : -self.w_halo + 1,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo,
        ]
        zm = self.raw[
            ...,
            self.w_halo - 1 : -self.w_halo,
            self.w_halo : -self.w_halo,
            self.w_halo : -self.w_halo,
        ]
        xf = 0.5 * (xp + xm)
        yf = 0.5 * (yp + ym)
        zf = 0.5 * (zp + zm)
        return FaceTensor(self.w_interior, xf, yf, zf)


def grad(field: CellTensor, dx: Float[torch.Tensor, " 3"]) -> FaceTensor:
    xp = field.raw[
        ...,
        field.w_halo : -field.w_halo,
        field.w_halo : -field.w_halo,
        field.w_halo : -field.w_halo + 1,
    ]
    xm = field.raw[
        ...,
        field.w_halo : -field.w_halo,
        field.w_halo : -field.w_halo,
        field.w_halo - 1 : -field.w_halo,
    ]
    yp = field.raw[
        ...,
        field.w_halo : -field.w_halo,
        field.w_halo : -field.w_halo + 1,
        field.w_halo : -field.w_halo,
    ]
    ym = field.raw[
        ...,
        field.w_halo : -field.w_halo,
        field.w_halo - 1 : -field.w_halo,
        field.w_halo : -field.w_halo,
    ]
    zp = field.raw[
        ...,
        field.w_halo : -field.w_halo + 1,
        field.w_halo : -field.w_halo,
        field.w_halo : -field.w_halo,
    ]
    zm = field.raw[
        ...,
        field.w_halo - 1 : -field.w_halo,
        field.w_halo : -field.w_halo,
        field.w_halo : -field.w_halo,
    ]
    ddx = (xp - xm) / dx[0]
    ddy = (yp - ym) / dx[1]
    ddz = (zp - zm) / dx[2]
    return FaceTensor(field.w_interior, ddx, ddy, ddz)

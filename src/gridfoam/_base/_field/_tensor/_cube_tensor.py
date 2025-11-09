from __future__ import annotations

import torch
from jaxtyping import Float


class CubeTensor:
    """
    A tensor representing cube-centered data.

    This class provides a structured way to handle 3D cube-centered data
    for finite volume methods.
    """

    def __init__(
        self,
        C: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """
        Initialize a CubeTensor with the specified parameters.

        Parameters
        ----------
        C : int
            Number of channels.
        dtype : torch.dtype
            Data type of the tensor.
        device : torch.device
            Device where the tensor will be allocated.

        Returns
        -------
        CubeTensor
            Initialized CellTensor instance.
        """
        self._C = C
        self._raw = torch.zeros((C,), dtype=dtype, device=device)

    @classmethod
    def zeros_like(cls, other: CubeTensor) -> CubeTensor:
        return cls(other.C, other.raw.dtype, other.raw.device)

    def __add__(self, other: CubeTensor) -> CubeTensor:
        cube_tensor = CubeTensor.zeros_like(self)
        cube_tensor.raw = self.raw + other.raw
        return cube_tensor

    def __sub__(self, other: CubeTensor) -> CubeTensor:
        cube_tensor = CubeTensor.zeros_like(self)
        cube_tensor.raw = self.raw - other.raw
        return cube_tensor

    def __mul__(self, other: CubeTensor) -> CubeTensor:
        cube_tensor = CubeTensor.zeros_like(self)
        cube_tensor.raw = self.raw * other.raw
        return cube_tensor

    @property
    def C(self) -> int:
        return self._C

    @property
    def raw(self) -> Float[torch.Tensor, " C"]:
        return self._raw

    @raw.setter
    def raw(self, value: Float[torch.Tensor, " C"]) -> None:
        self._raw = value

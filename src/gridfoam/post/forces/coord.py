from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float

from gridfoam.meta.config import DragLiftCoord, DragPitchCoord, ForceCoord
from gridfoam.meta.enums import ForceCoordMode


def normalize_vector(
    arr: Float[torch.Tensor, " 3"],
) -> Float[torch.Tensor, " 3"]:
    norm = torch.linalg.vector_norm(arr)
    if norm == 0:
        raise ValueError("vector is zero vector")
    return arr / norm


@dataclass(frozen=True)
class OrthonormalCoord:
    """
    Orthonormal basis used for force and moment coefficients.

    Attributes
    ----------
    e1 : torch.Tensor
        Drag and roll axis.
    e2 : torch.Tensor
        Side-force and pitch axis.
    e3 : torch.Tensor
        Lift and yaw axis.
    """

    e1: Float[torch.Tensor, " 3"]  # drag / roll
    e2: Float[torch.Tensor, " 3"]  # side / pitch
    e3: Float[torch.Tensor, " 3"]  # lift / yaw

    @classmethod
    def from_local_coord(cls, local_coord: ForceCoord) -> OrthonormalCoord:
        match local_coord.mode:
            case ForceCoordMode.DRAG_LIFT:
                assert isinstance(local_coord, DragLiftCoord)
                e1 = normalize_vector(torch.tensor(local_coord.drag_dir))
                lift = torch.tensor(local_coord.lift_dir)
                e3_raw = lift - e1 * (lift @ e1)
                e3 = normalize_vector(e3_raw)
                e2 = torch.linalg.cross(e3, e1)

            case ForceCoordMode.DRAG_PITCH:
                assert isinstance(local_coord, DragPitchCoord)
                e1 = normalize_vector(torch.tensor(local_coord.drag_dir))
                pitch = torch.tensor(local_coord.pitch_axis)
                e2_raw = pitch - e1 * (pitch @ e1)
                e2 = normalize_vector(e2_raw)
                e3 = torch.linalg.cross(e1, e2)
        return cls(
            e1=e1,
            e2=e2,
            e3=e3,
        )

    def on_device(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> OrthonormalCoord:
        return OrthonormalCoord(
            e1=self.e1.to(dtype=dtype, device=device),
            e2=self.e2.to(dtype=dtype, device=device),
            e3=self.e3.to(dtype=dtype, device=device),
        )

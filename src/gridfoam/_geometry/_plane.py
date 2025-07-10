from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam._geometry._interface import IPlane


class Plane(IPlane):
    """
    Plane
    Attributes:
        n: normal vector
        d: distance in the direction of n from the origin, \
            i.e. d = dot(n, p) where p is a point on the plane
    """

    def __init__(
        self, normal: Float[torch.Tensor, " space_dim"], distance: float
    ):
        self._normal = normal
        self._distance = distance
        self._space_dim = normal.shape[0]

    @classmethod
    def from_normal_and_distance(
        cls, n: Float[torch.Tensor, " space_dim"], d: float
    ) -> Plane:
        norm = torch.norm(n)
        if norm == 0.0:
            raise ValueError("normal vector cannot be zero")
        normal = n / norm
        return cls(normal, d)

    @classmethod
    def from_normal_and_point(
        cls,
        n: Float[torch.Tensor, " space_dim"],
        p: Float[torch.Tensor, " space_dim"],
    ) -> Plane:
        norm = torch.norm(n)
        if norm == 0.0:
            raise ValueError("normal vector cannot be zero")
        normal = n / norm
        distance = torch.dot(normal, p).item()
        return cls(normal, distance)

    @property
    def space_dim(self) -> int:
        return self._space_dim

    @property
    def normal(self) -> Float[torch.Tensor, " space_dim"]:
        return self._normal

    @property
    def distance(self) -> float:
        return self._distance

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import fluxel
import graphlow as gl
import torch
from jaxtyping import Bool, Float, Int

from gridfoam.meta.config import SimulatorConfig
from gridfoam.meta.enums import DomainBoundaryPatch

if TYPE_CHECKING:
    from gridfoam.core.field import CellField, FaceField


class IGridBase(ABC):
    @abstractmethod
    def __init__(
        self,
        simulator_config: SimulatorConfig,
        fluxel_mesh: fluxel.ICfdMesh,
    ):
        pass

    @abstractmethod
    def register_cellfield(self, field: CellField):
        pass

    @abstractmethod
    def register_facefield(self, field: FaceField):
        pass

    @abstractmethod
    def get_cellfield(self, name: str) -> CellField | None:
        pass

    @abstractmethod
    def get_facefield(self, name: str) -> FaceField | None:
        pass

    @abstractmethod
    def cellfield_names(self) -> Iterator[str]:
        pass

    @abstractmethod
    def facefield_names(self) -> Iterator[str]:
        pass

    @abstractmethod
    def get_domain_bnd_mask(
        self, patch_name: DomainBoundaryPatch
    ) -> Bool[torch.Tensor, " F"]:
        pass

    @property
    @abstractmethod
    def surface_mesh(self) -> gl.TensorMesh[Any]:
        pass

    @property
    @abstractmethod
    def sim_config(self) -> SimulatorConfig:
        pass

    @property
    @abstractmethod
    def dt(self) -> float:
        pass

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abstractmethod
    def num_cells(self) -> int:
        pass

    @property
    @abstractmethod
    def num_internal_faces(self) -> int:
        pass

    @property
    @abstractmethod
    def num_domain_bnd_faces(self) -> int:
        pass

    @property
    @abstractmethod
    def patch_name_to_id(self) -> dict[str, int]:
        pass

    @property
    @abstractmethod
    def owner(self) -> Int[torch.Tensor, " F"]:
        pass

    @property
    @abstractmethod
    def neighbour(self) -> Int[torch.Tensor, " F"]:
        pass

    @property
    @abstractmethod
    def axis(self) -> Int[torch.Tensor, " F"]:
        pass

    @property
    @abstractmethod
    def domain_bnd_owner(self) -> Int[torch.Tensor, " F_bnd"]:
        pass

    @property
    @abstractmethod
    def domain_bnd_dir_id(self) -> Int[torch.Tensor, " F_bnd"]:
        pass

    @property
    @abstractmethod
    def cell_centers(self) -> Float[torch.Tensor, " C 3"]:
        pass

    @property
    @abstractmethod
    def cell_sizes(self) -> Float[torch.Tensor, " C 3"]:
        pass

    @property
    @abstractmethod
    def cell_volumes(self) -> Float[torch.Tensor, " C"]:
        pass

    @property
    @abstractmethod
    def face_centers(self) -> Float[torch.Tensor, " F 3"]:
        pass

    @property
    @abstractmethod
    def Sf(self) -> Float[torch.Tensor, "F 3"]:
        pass

    @property
    @abstractmethod
    def domain_bnd_face_centers(self) -> Float[torch.Tensor, " F_bnd 3"]:
        pass

    @property
    @abstractmethod
    def domain_bnd_Sf(self) -> Float[torch.Tensor, "F_bnd 3"]:
        pass

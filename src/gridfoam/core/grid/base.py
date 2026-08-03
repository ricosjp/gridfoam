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
    """
    Abstract computational-grid interface shared by all mesh backends.

    Attributes
    ----------
    sim_config : SimulatorConfig
        Simulator configuration associated with this grid.
    dt : float
        Current time-step size.
    dtype : torch.dtype
        Floating-point dtype used for field tensors.
    device : torch.device
        Device used for field tensors.
    num_cells : int
        Number of cells ``C``.
    num_internal_faces : int
        Number of internal faces ``F``.
    num_domain_bnd_faces : int
        Number of domain-boundary faces ``F_bnd``.
    patch_name_to_id : dict[str, int]
        Mapping from patch name to patch id.
    owner : torch.Tensor
        Owner cell indices for internal faces, shape ``[F]``.
    neighbour : torch.Tensor
        Neighbour cell indices for internal faces, shape ``[F]``.
    axis : torch.Tensor
        Cartesian axis id for each internal face, shape ``[F]``.
    domain_bnd_owner : torch.Tensor
        Owner cell indices for domain-boundary faces, shape ``[F_bnd]``.
    domain_bnd_dir_id : torch.Tensor
        Outward direction id for domain-boundary faces, shape ``[F_bnd]``.
    cell_centers : torch.Tensor
        Cell centers with shape ``[C, 3]``.
    cell_sizes : torch.Tensor
        Cell edge lengths with shape ``[C, 3]``.
    cell_volumes : torch.Tensor
        Cell volumes with shape ``[C]``.
    face_centers : torch.Tensor
        Internal-face centers with shape ``[F, 3]``.
    Sf : torch.Tensor
        Internal-face area vectors with shape ``[F, 3]``.
    domain_bnd_face_centers : torch.Tensor
        Domain-boundary face centers with shape ``[F_bnd, 3]``.
    domain_bnd_Sf : torch.Tensor
        Domain-boundary face area vectors with shape ``[F_bnd, 3]``.
    surface_mesh : graphlow.TensorMesh
        Surface mesh used for force and visualization sampling.
    """

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
        """Surface mesh used for force and visualization sampling."""
        pass

    @property
    @abstractmethod
    def sim_config(self) -> SimulatorConfig:
        """Simulator configuration associated with this grid."""
        pass

    @property
    @abstractmethod
    def dt(self) -> float:
        """Current time-step size."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Floating-point dtype used for field tensors."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device used for field tensors."""
        pass

    @property
    @abstractmethod
    def num_cells(self) -> int:
        """Number of cells ``C``."""
        pass

    @property
    @abstractmethod
    def num_internal_faces(self) -> int:
        """Number of internal faces ``F``."""
        pass

    @property
    @abstractmethod
    def num_domain_bnd_faces(self) -> int:
        """Number of domain-boundary faces ``F_bnd``."""
        pass

    @property
    @abstractmethod
    def patch_name_to_id(self) -> dict[str, int]:
        """Mapping from patch name to patch id."""
        pass

    @property
    @abstractmethod
    def owner(self) -> Int[torch.Tensor, " F"]:
        """Owner cell indices for internal faces, shape ``[F]``."""
        pass

    @property
    @abstractmethod
    def neighbour(self) -> Int[torch.Tensor, " F"]:
        """Neighbour cell indices for internal faces, shape ``[F]``."""
        pass

    @property
    @abstractmethod
    def axis(self) -> Int[torch.Tensor, " F"]:
        """Cartesian axis id for each internal face, shape ``[F]``."""
        pass

    @property
    @abstractmethod
    def domain_bnd_owner(self) -> Int[torch.Tensor, " F_bnd"]:
        """Owner cell indices for domain-boundary faces, shape ``[F_bnd]``."""
        pass

    @property
    @abstractmethod
    def domain_bnd_dir_id(self) -> Int[torch.Tensor, " F_bnd"]:
        """Outward direction id for domain-boundary faces, shape ``[F_bnd]``."""
        pass

    @property
    @abstractmethod
    def cell_centers(self) -> Float[torch.Tensor, " C 3"]:
        """Cell centers with shape ``[C, 3]``."""
        pass

    @property
    @abstractmethod
    def cell_sizes(self) -> Float[torch.Tensor, " C 3"]:
        """Cell edge lengths with shape ``[C, 3]``."""
        pass

    @property
    @abstractmethod
    def cell_volumes(self) -> Float[torch.Tensor, " C"]:
        """Cell volumes with shape ``[C]``."""
        pass

    @property
    @abstractmethod
    def face_centers(self) -> Float[torch.Tensor, " F 3"]:
        """Internal-face centers with shape ``[F, 3]``."""
        pass

    @property
    @abstractmethod
    def Sf(self) -> Float[torch.Tensor, "F 3"]:
        """Internal-face area vectors with shape ``[F, 3]``."""
        pass

    @property
    @abstractmethod
    def domain_bnd_face_centers(self) -> Float[torch.Tensor, " F_bnd 3"]:
        """Domain-boundary face centers with shape ``[F_bnd, 3]``."""
        pass

    @property
    @abstractmethod
    def domain_bnd_Sf(self) -> Float[torch.Tensor, "F_bnd 3"]:
        """Domain-boundary face area vectors with shape ``[F_bnd, 3]``."""
        pass

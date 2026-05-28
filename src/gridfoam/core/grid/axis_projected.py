from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

import graphlow as gl
import torch
from fluxel import CfdAxisProjectedMesh
from jaxtyping import Bool, Float, Int

from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.config import SimulatorConfig
from gridfoam.meta.enums import DomainBoundaryPatch

if TYPE_CHECKING:
    from gridfoam.core.field import GeometricField
else:
    GeometricField = Any

logger = logging.getLogger(__name__)


class AxisProjectedGrid(IGridBase):
    def __init__(
        self,
        simulator_config: SimulatorConfig,
        fluxel_mesh: CfdAxisProjectedMesh,
        mesh_path: pathlib.Path | None = None,
    ):
        self._sim_config = simulator_config
        self._fields = WeakValueDictionary[str, GeometricField]()
        self._builtin_fields = WeakValueDictionary[str, GeometricField]()
        self._mesh_path = mesh_path
        self._surface_mesh_cache: gl.TensorMesh[Any] | None = None

        mesh = fluxel_mesh
        self._num_cells = mesh.n_cells
        self._num_internal_faces = mesh.internal_faces_owner.shape[0]
        self._num_domain_bnd_faces = mesh.domain_bnd_faces_owner.shape[0]
        self._num_immersed_faces: int = mesh.ap_is_immersed_face.sum().item()
        self._patch_name_to_id = deepcopy(mesh.patch_name_to_id)

        # ================================
        # Indices
        # ================================
        # Internal Faces
        self._owner = torch.from_numpy(mesh.internal_faces_owner).to(
            device=self.device, dtype=torch.long
        )
        self._neighbour = torch.from_numpy(mesh.internal_faces_neighbour).to(
            device=self.device, dtype=torch.long
        )
        self._axis = torch.from_numpy(mesh.internal_faces_axis).to(
            device=self.device, dtype=torch.long
        )

        # Domain Boundary Faces
        self._domain_bnd_owner = torch.from_numpy(
            mesh.domain_bnd_faces_owner
        ).to(device=self.device, dtype=torch.long)
        self._domain_bnd_dir_id = torch.from_numpy(
            mesh.domain_bnd_faces_dir
        ).to(device=self.device, dtype=torch.long)

        # Immersed Boundary Faces Mask [F,]
        self._ap_is_immersed_faces = torch.from_numpy(
            mesh.ap_is_immersed_face
        ).to(device=self.device, dtype=torch.bool)

        # Immersed Boundary Owner Patch IDs [F_immersed]
        self._ap_owner_bnd_patch_id = torch.from_numpy(
            mesh.ap_owner_bnd_patch_id
        ).to(device=self.device, dtype=torch.long)

        # Immersed Boundary Neighbour Patch IDs [F_immersed]
        self._ap_neighbour_bnd_patch_id = torch.from_numpy(
            mesh.ap_neighbour_bnd_patch_id
        ).to(device=self.device, dtype=torch.long)

        # Immersed Boundary Owner Face Anchor IDs [F_immersed]
        self._ap_owner_bnd_anchor_id = torch.from_numpy(
            mesh.ap_owner_bnd_anchor_id
        ).to(device=self.device, dtype=torch.long)

        # Immersed Boundary Neighbour Face Anchor IDs [F_immersed]
        self._ap_neighbour_bnd_anchor_id = torch.from_numpy(
            mesh.ap_neighbour_bnd_anchor_id
        ).to(device=self.device, dtype=torch.long)

        # ================================
        # Geometry
        # ================================
        # Cell Centers
        self._cell_centers = torch.from_numpy(mesh.cell_centers).to(
            device=self.device,
            dtype=self.dtype,
        )
        self._cell_sizes = torch.from_numpy(mesh.cell_sizes).to(
            device=self.device,
            dtype=self.dtype,
        )

        # Cell Volumes
        self._cell_volumes = torch.prod(self._cell_sizes, dim=1, keepdim=True)

        # Internal Face Centers
        self._face_centers = _compute_face_centers(
            self._cell_sizes, self._cell_centers, self._owner, self._neighbour
        )

        # Internal Face Surface Areas
        self._Sf = _compute_internal_Sf(
            self._cell_sizes, self._owner, self._neighbour, self._axis
        )

        # Domain Boundary Face Centers
        self._domain_bnd_face_centers = _compute_domain_bnd_face_centers(
            self._cell_sizes,
            self._cell_centers,
            self._domain_bnd_owner,
            self._domain_bnd_dir_id,
        )

        # Domain Boundary Face Surface Areas
        self._domain_bnd_Sf = _compute_domain_bnd_Sf(
            self._cell_sizes, self._domain_bnd_owner, self._domain_bnd_dir_id
        )

        # Immersed Boundary distances from owner to boundary [F_immersed 1]
        self._ap_dist_owner_to_bnd = (
            torch.from_numpy(mesh.ap_dist_owner_to_bnd)
            .to(device=self.device, dtype=self.dtype)
            .reshape(-1, 1)
        )
        # Immersed Boundary Owner Weights [F_immersed 2]
        self._ap_owner_weights = (
            torch.from_numpy(mesh.ap_owner_weights)
            .to(device=self.device, dtype=self.dtype)
            .reshape(-1, 2)
        )
        # Immersed Boundary distances from neighbour to boundary [F_immersed 1]
        self._ap_dist_neighbour_to_bnd = (
            torch.from_numpy(mesh.ap_dist_neighbour_to_bnd)
            .to(device=self.device, dtype=self.dtype)
            .reshape(-1, 1)
        )
        # Immersed Boundary Neighbour Weights [F_immersed 2]
        self._ap_neighbour_weights = (
            torch.from_numpy(mesh.ap_neighbour_weights)
            .to(device=self.device, dtype=self.dtype)
            .reshape(-1, 2)
        )

    def register_field(self, field: GeometricField):
        self._fields[field.name] = field

    def get_field(self, name: str) -> GeometricField | None:
        """
        Return a field associated with this grid
        for boundary-condition reference.

        This accessor is read-only by intent. Use it only to read existing
        field values in boundary conditions (e.g. InletOutlet,
        FixedFluxPressure), and do not use it as an update path for Field data.
        """
        return self._fields.get(name)

    def field_names(self) -> Iterator[str]:
        return self._fields.keys()

    def register_builtin_field(self, key: str, field: GeometricField):
        self._builtin_fields[key] = field

    def get_builtin_field(self, key: str) -> GeometricField | None:
        return self._builtin_fields.get(key)

    def builtin_field_keys(self) -> Iterator[str]:
        return self._builtin_fields.keys()

    def get_domain_bnd_mask(
        self, patch_name: DomainBoundaryPatch
    ) -> Bool[torch.Tensor, " F"]:
        direction = patch_name.to_direction()
        return self._domain_bnd_dir_id == direction.value

    def ap_get_patch_mask(
        self, patch_name: str
    ) -> tuple[Bool[torch.Tensor, " F"], Bool[torch.Tensor, " F"]]:
        patch_id = self._patch_name_to_id.get(patch_name, None)
        if patch_id is None:
            logger.warning(
                "patch name: %s is not found. skipping boundary condition "
                "evaluation.",
                patch_name,
            )
            none_mask = torch.zeros(
                self.num_immersed_faces, dtype=torch.bool, device=self.device
            )
            return none_mask, none_mask
        upper_mask = self._ap_owner_bnd_patch_id == patch_id
        lower_mask = self._ap_neighbour_bnd_patch_id == patch_id
        return upper_mask, lower_mask

    @property
    def sim_config(self) -> SimulatorConfig:
        return self._sim_config

    @property
    def surface_mesh(self) -> gl.TensorMesh[Any]:
        if self._mesh_path is None:
            raise ValueError("surface_mesh requires a mesh_path. ")
        if self._surface_mesh_cache is None:
            self._surface_mesh_cache = gl.read(
                str(self._mesh_path),
                backend="torch",
                dtype=self.dtype,
                device=self.device,
            )
        return self._surface_mesh_cache

    @property
    def dt(self) -> float:
        return self._sim_config.control.deltaT

    @property
    def dtype(self) -> torch.dtype:
        return self._sim_config.control.precision.to_torch_dtype()

    @property
    def device(self) -> torch.device:
        return self._sim_config.device.to_torch_device()

    @property
    def num_cells(self) -> int:
        return self._num_cells

    @property
    def num_internal_faces(self) -> int:
        return self._num_internal_faces

    @property
    def num_domain_bnd_faces(self) -> int:
        return self._num_domain_bnd_faces

    @property
    def num_immersed_faces(self) -> int:
        return self._num_immersed_faces

    @property
    def patch_name_to_id(self) -> dict[str, int]:
        return self._patch_name_to_id

    @property
    def owner(self) -> Int[torch.Tensor, " F"]:
        return self._owner

    @property
    def neighbour(self) -> Int[torch.Tensor, " F"]:
        return self._neighbour

    @property
    def axis(self) -> Int[torch.Tensor, " F"]:
        return self._axis

    @property
    def domain_bnd_owner(self) -> Int[torch.Tensor, " F_bnd"]:
        return self._domain_bnd_owner

    @property
    def domain_bnd_dir_id(self) -> Int[torch.Tensor, " F_bnd"]:
        return self._domain_bnd_dir_id

    @property
    def cell_centers(self) -> Float[torch.Tensor, " C 3"]:
        return self._cell_centers

    @property
    def cell_sizes(self) -> Float[torch.Tensor, " C 3"]:
        return self._cell_sizes

    @property
    def cell_volumes(self) -> Float[torch.Tensor, " C 1"]:
        return self._cell_volumes

    @property
    def face_centers(self) -> Float[torch.Tensor, " F 3"]:
        return self._face_centers

    @property
    def Sf(self) -> Float[torch.Tensor, "F 3"]:
        return self._Sf

    @property
    def domain_bnd_face_centers(self) -> Float[torch.Tensor, " F_bnd 3"]:
        return self._domain_bnd_face_centers

    @property
    def domain_bnd_Sf(self) -> Float[torch.Tensor, "F_bnd 3"]:
        return self._domain_bnd_Sf

    @property
    def ap_is_immersed_faces(self) -> Bool[torch.Tensor, " F"]:
        return self._ap_is_immersed_faces

    @property
    def ap_owner_bnd_patch_id(self) -> Int[torch.Tensor, " F_immersed"]:
        return self._ap_owner_bnd_patch_id

    @property
    def ap_neighbour_bnd_patch_id(self) -> Int[torch.Tensor, " F_immersed"]:
        return self._ap_neighbour_bnd_patch_id

    @property
    def ap_dist_owner_to_bnd(self) -> Float[torch.Tensor, " F_immersed 1"]:
        return self._ap_dist_owner_to_bnd

    @property
    def ap_owner_weights(self) -> Float[torch.Tensor, " F_immersed 2"]:
        return self._ap_owner_weights

    @property
    def ap_dist_neighbour_to_bnd(self) -> Float[torch.Tensor, " F_immersed 1"]:
        return self._ap_dist_neighbour_to_bnd

    @property
    def ap_neighbour_weights(self) -> Float[torch.Tensor, " F_immersed 2"]:
        return self._ap_neighbour_weights

    @property
    def ap_owner_bnd_anchor_id(self) -> Int[torch.Tensor, " F_immersed"]:
        return self._ap_owner_bnd_anchor_id

    @property
    def ap_neighbour_bnd_anchor_id(self) -> Int[torch.Tensor, " F_immersed"]:
        return self._ap_neighbour_bnd_anchor_id


def _compute_face_centers(
    cell_sizes: Float[torch.Tensor, " C 3"],
    cell_centers: Float[torch.Tensor, " C 3"],
    owner: Int[torch.Tensor, " F"],
    neighbour: Int[torch.Tensor, " F"],
) -> Float[torch.Tensor, "F_internal 3"]:
    c_own = cell_centers[owner]
    c_nei = cell_centers[neighbour]
    s_own = cell_sizes[owner]
    s_nei = cell_sizes[neighbour]

    # For hanging-node octree interfaces, compute the shared-face overlap
    # region from owner and neighbor bounding boxes, then take its center.
    own_min = c_own - s_own / 2.0
    own_max = c_own + s_own / 2.0
    nei_min = c_nei - s_nei / 2.0
    nei_max = c_nei + s_nei / 2.0

    face_min = torch.maximum(own_min, nei_min)
    face_max = torch.minimum(own_max, nei_max)
    Cf = (face_min + face_max) / 2.0

    return Cf


def _compute_internal_Sf(
    cell_sizes: Float[torch.Tensor, " C 3"],
    owner: Int[torch.Tensor, " F"],
    neighbour: Int[torch.Tensor, " F"],
    axis: Int[torch.Tensor, " F"],
) -> Float[torch.Tensor, "F_internal 3"]:
    num_internal_faces = owner.shape[0]
    dtype = cell_sizes.dtype
    device = cell_sizes.device

    s_own = cell_sizes[owner]
    s_nei = cell_sizes[neighbour]

    # Use the finer-cell size on each interface.
    min_sizes = torch.minimum(s_own, s_nei)

    # Axis index normal to each face (N, 1).
    axis_idx = axis[:, None]

    # Face area = cell volume / normal-axis length.
    vol = torch.prod(min_sizes, dim=1, keepdim=True)
    axis_len = min_sizes.gather(1, axis_idx)
    area = vol / axis_len  # [F_internal 1]

    # Build face area vector Sf. In orthogonal grids, owner -> neighbor
    # is always positive. Scatter area only into the corresponding axis.
    Sf = torch.zeros((num_internal_faces, 3), dtype=dtype, device=device)
    return Sf.scatter_add_(1, axis_idx, area)


def _compute_domain_bnd_face_centers(
    cell_sizes: Float[torch.Tensor, " C 3"],
    cell_centers: Float[torch.Tensor, " C 3"],
    domain_bnd_owner: Int[torch.Tensor, " F_bnd"],
    domain_bnd_dir_id: Int[torch.Tensor, " F_bnd"],
) -> Float[torch.Tensor, "F_bnd 3"]:
    num_domain_bnd_faces = domain_bnd_owner.shape[0]
    dtype = cell_centers.dtype
    device = cell_centers.device

    normals = torch.zeros((num_domain_bnd_faces, 3), dtype=dtype, device=device)
    axis_idx = (domain_bnd_dir_id // 2)[:, None]
    sign = (2.0 * (domain_bnd_dir_id % 2) - 1.0).to(dtype=dtype)
    normals = normals.scatter_add_(1, axis_idx, sign[:, None])
    c_own = cell_centers[domain_bnd_owner]
    s_own = cell_sizes[domain_bnd_owner]
    return c_own + 0.5 * s_own * normals


def _compute_domain_bnd_Sf(
    cell_sizes: Float[torch.Tensor, " C 3"],
    domain_bnd_owner: Int[torch.Tensor, " F_bnd"],
    domain_bnd_dir_id: Int[torch.Tensor, " F_bnd"],
) -> Float[torch.Tensor, "F_bnd 3"]:
    num_domain_bnd_faces = domain_bnd_owner.shape[0]
    dtype = cell_sizes.dtype
    device = cell_sizes.device

    s_own = cell_sizes[domain_bnd_owner]

    sign = (2.0 * (domain_bnd_dir_id % 2) - 1.0).to(dtype=dtype)
    axis_idx = (domain_bnd_dir_id // 2)[:, None]

    vol = torch.prod(s_own, dim=1, keepdim=True)
    axis_len = s_own.gather(1, axis_idx)
    area = sign[:, None] * vol / axis_len  # [F_bnd 1]

    # Build face area vector Sf by scattering area to the axis column.
    Sf = torch.zeros((num_domain_bnd_faces, 3), dtype=dtype, device=device)
    return Sf.scatter_add_(1, axis_idx, area)

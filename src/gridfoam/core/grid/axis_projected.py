from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Self
from weakref import WeakValueDictionary

import graphlow as gl
import numpy as np
import torch
from fluxel import (
    ApibmSession,
    Axis,
    CfdAxisProjectedMesh,
    quaternion_from_axis_angle,
)
from jaxtyping import Bool, Float, Int

from gridfoam.core.fv_cache import FvGridCache
from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.config import SimulatorConfig
from gridfoam.meta.enums import DomainBoundaryPatch

if TYPE_CHECKING:
    from gridfoam.core.field import CellField, FaceField
else:
    CellField = Any
    FaceField = Any

logger = logging.getLogger(__name__)

_DEVICE_TENSOR_ATTRS = (
    "_owner",
    "_neighbour",
    "_axis",
    "_domain_bnd_owner",
    "_domain_bnd_dir_id",
    "_cell_centers",
    "_cell_sizes",
    "_cell_volumes",
    "_face_centers",
    "_Sf",
    "_domain_bnd_face_centers",
    "_domain_bnd_Sf",
    "_ap_is_immersed_faces",
    "_ap_owner_bnd_patch_id",
    "_ap_neighbour_bnd_patch_id",
    "_ap_owner_bnd_anchor_id",
    "_ap_neighbour_bnd_anchor_id",
    "_ap_dist_owner_to_bnd",
    "_ap_owner_weights",
    "_ap_dist_neighbour_to_bnd",
    "_ap_neighbour_weights",
)


class AxisProjectedGrid(IGridBase):
    """
    Axis-projected immersed-boundary grid.

    Static meshes are built once from a fluxel CFD snapshot. Dynamic meshes
    keep an :class:`fluxel.ApibmSession` so :meth:`update_ib` and
    :meth:`remesh` can follow a moving boundary.
    """

    def __init__(
        self,
        simulator_config: SimulatorConfig,
        fluxel_mesh: CfdAxisProjectedMesh,
        mesh_path: pathlib.Path | None = None,
        session: ApibmSession | None = None,
    ):
        self._sim_config = simulator_config
        self._runtime_device: torch.device | None = None
        self._session = session
        self._cellfields = WeakValueDictionary[str, CellField]()
        self._facefields = WeakValueDictionary[str, FaceField]()
        self._mesh_path = mesh_path
        self._surface_mesh_cache: gl.TensorMesh[Any] | None = None
        self._surface_mesh_rest_points: torch.Tensor | None = None
        self._fv_cache = FvGridCache()

        self._load_topology(fluxel_mesh)
        self._load_ap_payload(fluxel_mesh)

    def _require_session(self) -> ApibmSession:
        """Return the fluxel session, or raise if this grid is static."""
        if self._session is None:
            raise RuntimeError(
                "IBM updates require fluxel.motion=dynamic so the grid is "
                "built with an ApibmSession."
            )
        return self._session

    def _load_topology(self, mesh: CfdAxisProjectedMesh) -> None:
        """Load background mesh topology and derived geometry tensors."""
        self._num_cells = mesh.n_cells
        self._num_internal_faces = mesh.internal_faces_owner.shape[0]
        self._num_domain_bnd_faces = mesh.domain_bnd_faces_owner.shape[0]
        self._patch_name_to_id = deepcopy(mesh.patch_name_to_id)

        # ================================
        # Indices
        # ================================
        # Internal Faces
        self._owner = _numpy_to_torch(
            mesh.internal_faces_owner, device=self.device, dtype=torch.long
        )
        self._neighbour = _numpy_to_torch(
            mesh.internal_faces_neighbour, device=self.device, dtype=torch.long
        )
        self._axis = _numpy_to_torch(
            mesh.internal_faces_axis, device=self.device, dtype=torch.long
        )

        # Domain Boundary Faces
        self._domain_bnd_owner = _numpy_to_torch(
            mesh.domain_bnd_faces_owner, device=self.device, dtype=torch.long
        )
        self._domain_bnd_dir_id = _numpy_to_torch(
            mesh.domain_bnd_faces_dir, device=self.device, dtype=torch.long
        )

        # ================================
        # Geometry
        # ================================
        # Cell Centers
        self._cell_centers = _numpy_to_torch(
            mesh.cell_centers, device=self.device, dtype=self.dtype
        )
        self._cell_sizes = _numpy_to_torch(
            mesh.cell_sizes, device=self.device, dtype=self.dtype
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

    def _load_ap_payload(self, mesh: CfdAxisProjectedMesh) -> None:
        """Load compressed axis-projected IBM arrays from ``mesh.ap``."""
        ap = mesh.ap

        # Immersed Boundary Faces Mask [F,]
        self._ap_is_immersed_faces = _numpy_to_torch(
            ap.is_immersed_face, device=self.device, dtype=torch.bool
        )
        self._num_immersed_faces = int(self._ap_is_immersed_faces.sum().item())

        # Immersed Boundary Owner Patch IDs [F_immersed]
        self._ap_owner_bnd_patch_id = _numpy_to_torch(
            ap.owner_bnd_patch_id, device=self.device, dtype=torch.long
        )

        # Immersed Boundary Neighbour Patch IDs [F_immersed]
        self._ap_neighbour_bnd_patch_id = _numpy_to_torch(
            ap.neighbour_bnd_patch_id, device=self.device, dtype=torch.long
        )

        # Immersed Boundary Owner Face Anchor IDs [F_immersed]
        self._ap_owner_bnd_anchor_id = _numpy_to_torch(
            ap.owner_bnd_anchor_id, device=self.device, dtype=torch.long
        )

        # Immersed Boundary Neighbour Face Anchor IDs [F_immersed]
        self._ap_neighbour_bnd_anchor_id = _numpy_to_torch(
            ap.neighbour_bnd_anchor_id, device=self.device, dtype=torch.long
        )

        # Immersed Boundary distances from owner to boundary [F_immersed 1]
        self._ap_dist_owner_to_bnd = _numpy_to_torch(
            ap.dist_owner_to_bnd, device=self.device, dtype=self.dtype
        ).reshape(-1, 1)

        # Immersed Boundary Owner Weights [F_immersed 2]
        self._ap_owner_weights = _numpy_to_torch(
            ap.owner_weights, device=self.device, dtype=self.dtype
        ).reshape(-1, 2)

        # Immersed Boundary distances from neighbour to boundary [F_immersed 1]
        self._ap_dist_neighbour_to_bnd = _numpy_to_torch(
            ap.dist_neighbour_to_bnd, device=self.device, dtype=self.dtype
        ).reshape(-1, 1)

        # Immersed Boundary Neighbour Weights [F_immersed 2]
        self._ap_neighbour_weights = _numpy_to_torch(
            ap.neighbour_weights, device=self.device, dtype=self.dtype
        ).reshape(-1, 2)

    def _sync_registered_fields(self, *, topology_changed: bool) -> None:
        """Resize registered fields after IBM or topology updates."""
        self.invalidate_derived_caches()
        for field in list(self._cellfields.values()):
            field.sync_to_grid_topology(topology_changed=topology_changed)
        for field in list(self._facefields.values()):
            field.sync_to_grid_topology(topology_changed=topology_changed)

    def _apply_ib_pose_to_surface_mesh(self) -> None:
        """Apply the current IB rigid pose to the cached surface mesh."""
        if (
            self._surface_mesh_cache is None
            or self._surface_mesh_rest_points is None
        ):
            return
        rest = self._surface_mesh_rest_points
        rotated = _rotate_points_by_quaternion(
            rest, self.ib_rotation_quaternion
        )
        translation = torch.tensor(
            self.ib_translation, dtype=rest.dtype, device=rest.device
        )
        posed = rotated + translation
        self._surface_mesh_cache.points = posed
        # TensorMesh.save writes pvmesh.points, not the tensor.
        self._surface_mesh_cache.pvmesh.points = posed.detach().cpu().numpy()

    def update_ib(
        self,
        translation: list[float] | None = None,
        rotation_axis: Axis | Sequence[float] | None = None,
        rotation_angle: float | None = None,
        *,
        degrees: bool = False,
        rotation_quaternion: list[float] | None = None,
        warn_outside_refinement: bool = True,
    ) -> AxisProjectedGrid:
        """
        Recompute immersed-boundary data with a fixed background topology.

        Pose arguments are absolute. Rotation is specified by
        ``rotation_axis`` and ``rotation_angle``. Pass
        ``rotation_quaternion`` only when a unit quaternion is already
        available. Omitted components keep the current pose. Registered
        face fields are resized to the new immersed-face set; cell fields
        are unchanged.

        Parameters
        ----------
        translation : list of float or None
            Absolute translation ``[tx, ty, tz]``.
        rotation_axis : fluxel.Axis or sequence of float or None
            Rotation axis. Pass ``Axis.X`` / ``Axis.Y`` / ``Axis.Z``, or a
            3-vector. Must be given together with ``rotation_angle``.
        rotation_angle : float or None
            Rotation angle. Radians by default; set ``degrees=True`` for
            degrees.
        degrees : bool, default False
            If True, ``rotation_angle`` is interpreted in degrees.
        rotation_quaternion : list of float or None
            Absolute unit quaternion ``[w, x, y, z]``. Keyword-only escape
            hatch, mutually exclusive with ``rotation_axis`` /
            ``rotation_angle``.
        warn_outside_refinement : bool, default True
            If True, warn when the IB intersects cells below the target level.

        Returns
        -------
        AxisProjectedGrid
            This grid after the IBM payload has been refreshed.
        """
        mesh = self._require_session().update_ib(
            translation=translation,
            rotation_quaternion=_resolve_rotation_quaternion(
                rotation_quaternion,
                rotation_axis,
                rotation_angle,
                degrees=degrees,
            ),
            warn_outside_refinement=warn_outside_refinement,
        )
        self._load_ap_payload(mesh)
        self._apply_ib_pose_to_surface_mesh()
        self._sync_registered_fields(topology_changed=False)
        return self

    def remesh(
        self,
        target_level: int | None = None,
        refinement_regions: (
            list[tuple[list[float], list[float], int]] | None
        ) = None,
        translation: list[float] | None = None,
        rotation_axis: Axis | Sequence[float] | None = None,
        rotation_angle: float | None = None,
        *,
        degrees: bool = False,
        rotation_quaternion: list[float] | None = None,
        warn_outside_refinement: bool = True,
    ) -> AxisProjectedGrid:
        """
        Rebuild the AMR background mesh and IBM payload for the current pose.

        Topology, geometry, and IBM arrays are replaced. Registered fields
        are reallocated to the new sizes; previous field values are discarded.
        Rotation is specified by ``rotation_axis`` and ``rotation_angle``.
        Pass ``rotation_quaternion`` only when a unit quaternion is already
        available.

        Parameters
        ----------
        target_level : int or None
            New maximum octree refinement level around the surface.
            ``None`` keeps the current target level.
        refinement_regions : list of tuple or None
            Optional region refinement requests as ``(min, max, level)``.
            ``None`` keeps the current region list.
        translation : list of float or None
            Absolute translation ``[tx, ty, tz]``.
        rotation_axis : fluxel.Axis or sequence of float or None
            Rotation axis. Pass ``Axis.X`` / ``Axis.Y`` / ``Axis.Z``, or a
            3-vector. Must be given together with ``rotation_angle``.
        rotation_angle : float or None
            Rotation angle. Radians by default; set ``degrees=True`` for
            degrees.
        degrees : bool, default False
            If True, ``rotation_angle`` is interpreted in degrees.
        rotation_quaternion : list of float or None
            Absolute unit quaternion ``[w, x, y, z]``. Keyword-only escape
            hatch, mutually exclusive with ``rotation_axis`` /
            ``rotation_angle``.
        warn_outside_refinement : bool, default True
            If True, warn when the IB intersects cells below the target level.

        Returns
        -------
        AxisProjectedGrid
            This grid after AMR and IBM reconstruction.
        """
        mesh = self._require_session().remesh(
            target_level=target_level,
            refinement_regions=refinement_regions,
            translation=translation,
            rotation_quaternion=_resolve_rotation_quaternion(
                rotation_quaternion,
                rotation_axis,
                rotation_angle,
                degrees=degrees,
            ),
            warn_outside_refinement=warn_outside_refinement,
        )
        self._load_topology(mesh)
        self._load_ap_payload(mesh)
        self._apply_ib_pose_to_surface_mesh()
        self._sync_registered_fields(topology_changed=True)
        return self

    def register_cellfield(self, field: CellField):
        self._cellfields[field.name] = field

    def register_facefield(self, field: FaceField):
        self._facefields[field.name] = field

    def get_cellfield(self, name: str) -> CellField | None:
        return self._cellfields.get(name)

    def get_facefield(self, name: str) -> FaceField | None:
        return self._facefields.get(name)

    def cellfield_names(self) -> Iterator[str]:
        return self._cellfields.keys()

    def facefield_names(self) -> Iterator[str]:
        return self._facefields.keys()

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
            self._surface_mesh_rest_points = (
                self._surface_mesh_cache.points.detach().clone()
            )
            self._apply_ib_pose_to_surface_mesh()
        return self._surface_mesh_cache

    @property
    def is_dynamic(self) -> bool:
        """Whether this grid keeps a fluxel session for IBM updates."""
        return self._session is not None

    @property
    def ib_translation(self) -> list[float]:
        """Current rigid translation ``[tx, ty, tz]`` applied to the IB mesh."""
        if self._session is None:
            return [0.0, 0.0, 0.0]
        return list(self._session.translation)

    @property
    def ib_rotation_quaternion(self) -> list[float]:
        """Current IB rotation as a unit quaternion ``[w, x, y, z]``."""
        if self._session is None:
            return [1.0, 0.0, 0.0, 0.0]
        return list(self._session.rotation_quaternion)

    @property
    def dt(self) -> float:
        return self._sim_config.control.deltaT

    @property
    def dtype(self) -> torch.dtype:
        return self._sim_config.control.precision.to_torch_dtype()

    @property
    def device(self) -> torch.device:
        if self._runtime_device is not None:
            return self._runtime_device
        return self._sim_config.device.to_torch_device()

    @property
    def fv_cache(self) -> FvGridCache:
        """Grid-owned FV derived-data cache."""
        return self._fv_cache

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        """
        Move geometry, IBM arrays, and registered fields to ``device``.

        The move is in-place. ``sim_config.device`` is left unchanged.

        Parameters
        ----------
        device : torch.device or str
            Target device.
        non_blocking : bool, default False
            Passed through to ``Tensor.to``.

        Returns
        -------
        AxisProjectedGrid
            This grid after tensors have been moved.
        """
        target = torch.device(device) if isinstance(device, str) else device
        if _devices_match(self.device, target):
            return self

        for name in _DEVICE_TENSOR_ATTRS:
            tensor = getattr(self, name)
            setattr(
                self,
                name,
                tensor.to(device=target, non_blocking=non_blocking),
            )

        self._runtime_device = self._owner.device
        self._surface_mesh_cache = None
        self._surface_mesh_rest_points = None
        self.invalidate_derived_caches()

        for field in list(self._cellfields.values()):
            field.to(target, non_blocking=non_blocking)
        for field in list(self._facefields.values()):
            field.to(target, non_blocking=non_blocking)
        return self

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


def _devices_match(left: torch.device, right: torch.device) -> bool:
    """Return whether two devices refer to the same runtime device."""
    if left.type != right.type:
        return False
    if left.type != "cuda":
        return True
    left_index = left.index
    right_index = right.index
    if left_index is None:
        left_index = torch.cuda.current_device()
    if right_index is None:
        right_index = torch.cuda.current_device()
    return left_index == right_index


def _resolve_rotation_quaternion(
    rotation_quaternion: list[float] | None,
    rotation_axis: Axis | Sequence[float] | None,
    rotation_angle: float | None,
    *,
    degrees: bool,
) -> list[float] | None:
    """Return a quaternion from explicit or axis-angle rotation arguments."""
    has_quaternion = rotation_quaternion is not None
    has_axis = rotation_axis is not None
    has_angle = rotation_angle is not None
    if has_quaternion and (has_axis or has_angle):
        raise ValueError(
            "Pass either rotation_quaternion or rotation_axis/"
            "rotation_angle, not both."
        )
    if has_axis != has_angle:
        raise ValueError(
            "rotation_axis and rotation_angle must be provided together."
        )
    if rotation_axis is not None and rotation_angle is not None:
        return quaternion_from_axis_angle(
            rotation_axis, rotation_angle, degrees=degrees
        )
    return rotation_quaternion


def _numpy_to_torch(
    array: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Copy a NumPy array onto the grid device and dtype."""
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def _rotate_points_by_quaternion(
    points: Float[torch.Tensor, " N 3"],
    quat: list[float],
) -> Float[torch.Tensor, " N 3"]:
    """
    Rotate points by a unit quaternion ``[w, x, y, z]``.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates with shape ``[N, 3]``.
    quat : list of float
        Unit quaternion ``[w, x, y, z]``.

    Returns
    -------
    torch.Tensor
        Rotated points with shape ``[N, 3]``.
    """
    w, x, y, z = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = torch.tensor(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=points.dtype,
        device=points.device,
    )
    return points @ rot.T


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

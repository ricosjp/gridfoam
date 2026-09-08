from __future__ import annotations

from abc import ABC, abstractmethod
from math import prod
from typing import Self

import torch
from jaxtyping import Bool, Float
from phlower_tensor import PhysicalDimensions

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.factory import create_boundary_condition
from gridfoam.core.dimensions import (
    DimensionLike,
    assert_compatible,
    resolve_field_dimension,
    to_dimensions,
)
from gridfoam.core.fv_cache import FvFieldCache
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import GridBase
from gridfoam.core.shapes import require_shape, validate_component_shape
from gridfoam.meta.config import TensorValue
from gridfoam.meta.enums import FieldRole
from gridfoam.meta.types import PatchName


def _tensor_token(tensor: torch.Tensor) -> tuple[int, int, int]:
    """Identity, storage pointer and in-place version of ``tensor``."""
    return (id(tensor), tensor.data_ptr(), tensor._version)


class GeometricField(ABC):
    """
    Abstract base class for geometric fields on a mesh.

    Holds common metadata such as name, grid, physical tensor shape, and role.

    Attributes
    ----------
    grid : GridBase
        Computational grid that owns this field.
    name : str
        Field name, optionally with a phase suffix.
    role : FieldRole
        Temporal role of the field.
    component_shape : tuple[int, ...]
        Physical axes: (), (3,), (3, 3), ...; excludes the entity axis.
    tensor_rank : int
        Physical tensor rank, ``len(component_shape)``.
    num_components : int
        Number of scalar entries in the physical tensor,
        ``prod(component_shape)``.
    dimension : PhysicalDimensions or None
        Optional physical dimension of the field.
    export : bool
        Whether the field is written to output.
    """

    _component_shape: tuple[int, ...]

    @property
    @abstractmethod
    def grid(self) -> GridBase:
        """Computational grid that owns this field."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Field name, optionally with a phase suffix."""
        pass

    @property
    @abstractmethod
    def role(self) -> FieldRole:
        """Temporal role of the field."""
        pass

    @property
    def component_shape(self) -> tuple[int, ...]:
        """Physical tensor axes, excluding the entity axis."""
        return self._component_shape

    @property
    def tensor_rank(self) -> int:
        """Physical tensor rank, ``len(component_shape)``."""
        return len(self.component_shape)

    @property
    def num_components(self) -> int:
        """Number of scalar entries in the physical tensor."""
        return prod(self.component_shape)

    @property
    @abstractmethod
    def dimension(self) -> PhysicalDimensions | None:
        """Optional physical dimension of the field."""
        pass

    @property
    @abstractmethod
    def export(self) -> bool:
        """Whether the field is written to output."""
        pass

    @abstractmethod
    def state_token(self) -> tuple[int, ...]:
        """
        Fingerprint of the current data used for cache invalidation.

        Returns
        -------
        tuple[int, ...]
            Changes whenever the field data is replaced or modified in place.
        """
        pass


class CellField(GeometricField):
    """
    Cell-centered field class.

    Attributes
    ----------
    grid : GridBase
        Computational grid that owns this field.
    name : str
        Field name, optionally with a phase suffix.
    role : FieldRole
        Temporal role of the field.
    component_shape : tuple[int, ...]
        Physical axes: (), (3,), (3, 3), ...; excludes the entity axis.
    tensor_rank : int
        Physical tensor rank, ``len(component_shape)``.
    num_components : int
        Number of scalar entries in the physical tensor,
        ``prod(component_shape)``.
    dimension : PhysicalDimensions or None
        Optional physical dimension of the field.
    export : bool
        Whether the field is written to output.
    bcs : dict[PatchName, BoundaryCondition]
        Boundary conditions keyed by patch name.
    data : torch.Tensor
        Cell values with shape ``[C, *component_shape]``.
    old_data : torch.Tensor
        Previous time-level values with shape ``[C, *component_shape]``.
    fv_cache : FvFieldCache
        Field-owned FV cache (boundary batches, states, and cell constraints).
    """

    def __init__(
        self,
        grid: GridBase,
        name: str,
        role: FieldRole,
        component_shape: tuple[int, ...],
        dimension: DimensionLike = None,
    ):
        self._grid = grid
        self._name = name
        self._role = role
        self._component_shape = validate_component_shape(component_shape)
        self._fv_cache = FvFieldCache()

        device = grid.device
        dtype = grid.dtype

        shape = (grid.num_cells, *self.component_shape)
        self._data = torch.zeros(shape, dtype=dtype, device=device)

        self._bcs: dict[PatchName, BoundaryCondition] = {}
        self._export = False
        self._condition = self.grid.sim_config.get_field_condition(self.name)
        config_dimension = (
            self._condition.dimension if self._condition is not None else None
        )
        self._dimension = resolve_field_dimension(
            self.name, explicit=dimension, config=config_dimension
        )
        # A caller-supplied dimension wins over the configured one, so reject
        # the combination instead of silently discarding the configuration.
        assert_compatible(
            self._dimension,
            to_dimensions(config_dimension),
            f"cell field {name!r} configuration",
        )
        if self._condition is not None:
            self._export = self._condition.export
            # init internal value
            self.reset_data(self._condition.internal)
            # init boundary conditions
            for bc_config in self._condition.iter_bc_configs():
                bc = create_boundary_condition(
                    bc_config, dtype=grid.dtype, device=grid.device
                )
                for patch in bc_config.patches:
                    self._bcs[patch] = bc

        # Self-register in the grid field registry.
        grid.register_cellfield(self)

        self.update_history(reset=True)

    def sync_to_grid_topology(self, *, topology_changed: bool = False) -> None:
        """
        Reallocate cell buffers when the grid cell count changes.

        Parameters
        ----------
        topology_changed : bool, default False
            If True, buffers are reallocated even when ``C`` is unchanged
            and previous values are discarded.
        """
        n_cells = self.grid.num_cells
        if (not topology_changed) and self._data.shape[0] == n_cells:
            # IBM updates reset face history even on a fixed background
            # mesh; restart cell history too so ddt and ddtCorr agree.
            self.update_history(reset=True)
            return
        self._data = torch.zeros(
            (n_cells, *self.component_shape),
            dtype=self.grid.dtype,
            device=self.grid.device,
        )
        self.fv_cache.clear()
        self.update_history(reset=True)

    def reset_data(self, init_value: TensorValue):
        """Reset using a uniform physical tensor, without the entity axis."""
        value = torch.tensor(
            init_value, dtype=self.grid.dtype, device=self.grid.device
        )
        require_shape(
            value, self.component_shape, f"Field {self.name!r} internal"
        )
        self._data[:] = value
        self.update_history(reset=True)

    def update_history(self, *, reset: bool = False):
        """
        Advance time history once per completed time step.

        Use ``reset=True`` after setting initial values or remapping a mesh;
        this discards the second old level and restarts BDF2 with Euler.
        """
        if self._role == FieldRole.TRANSIENT:
            self._older_data = None if reset else self._old_data
            self._previous_dt = None if reset else self.grid.dt
            self._old_data = self._data.clone()
            return
        self._older_data = None
        self._previous_dt = None
        self._old_data = self._data

    def add_boundary_conditions(self, bcs: dict[PatchName, BoundaryCondition]):
        """
        Add or update boundary conditions.
        """
        self._bcs.update(bcs)
        self.fv_cache.clear_boundary_batches()

    @property
    def fv_cache(self) -> FvFieldCache:
        """Cache of boundary batches, states, and cell constraints."""
        return self._fv_cache

    # ================================
    # Grid Accessors
    # ================================
    @property
    def grid(self) -> GridBase:
        """Computational grid that owns this field."""
        return self._grid

    # ================================
    # Metadata Accessors
    # ================================
    @property
    def name(self) -> str:
        """Field name, optionally with a phase suffix."""
        return self._name

    @property
    def role(self) -> FieldRole:
        """Temporal role of the field."""
        return self._role

    @property
    def dimension(self) -> PhysicalDimensions | None:
        """Optional physical dimension of the field."""
        return self._dimension

    @property
    def export(self) -> bool:
        """Whether the field is written to output."""
        return self._export

    @export.setter
    def export(self, value: bool):
        self._export = value

    @property
    def bcs(self) -> dict[PatchName, BoundaryCondition]:
        """Boundary conditions keyed by patch name."""
        return self._bcs

    # ================================
    # Data Accessors
    # ================================
    @property
    def data(self) -> Float[torch.Tensor, " C *component_shape"]:
        """Cell values with shape ``[C, *component_shape]``."""
        return self._data

    @data.setter
    def data(self, value: Float[torch.Tensor, " C *component_shape"]):
        require_shape(
            value,
            (self.grid.num_cells, *self.component_shape),
            f"{self.name}.data",
        )
        self._data = value

    @property
    def old_data(self) -> Float[torch.Tensor, " C *component_shape"]:
        """Previous time-level values with shape ``[C, *component_shape]``."""
        return self._old_data

    @property
    def older_data(self) -> torch.Tensor | None:
        """Second old time level, or None after initialization/remapping."""
        return self._older_data

    @property
    def previous_dt(self) -> float | None:
        """Time interval between the two stored old levels."""
        return self._previous_dt

    def state_token(self) -> tuple[int, ...]:
        """
        Fingerprint of the current cell data used for cache invalidation.

        Combines the tensor identity, storage pointer and in-place version
        counter, so both ``field.data = new`` and ``field.data[:] = x``
        change the token.
        """
        return _tensor_token(self._data)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        """
        Move cell buffers to ``device`` in place.

        Parameters
        ----------
        device : torch.device or str
            Target device.
        non_blocking : bool, default False
            Passed through to ``Tensor.to``.

        Returns
        -------
        CellField
            This field after the buffers have been moved.
        """
        old_aliased = self._old_data is self._data
        self._data = self._data.to(device=device, non_blocking=non_blocking)
        if old_aliased:
            self._old_data = self._data
        else:
            self._old_data = self._old_data.to(
                device=device, non_blocking=non_blocking
            )
        if self._older_data is not None:
            self._older_data = self._older_data.to(
                device=device, non_blocking=non_blocking
            )
        return self


class FaceField(GeometricField):
    """
    Face-centered field class.

    Attributes
    ----------
    grid : GridBase
        Computational grid that owns this field.
    name : str
        Field name, optionally with a phase suffix.
    role : FieldRole
        Temporal role of the field.
    component_shape : tuple[int, ...]
        Physical axes: (), (3,), (3, 3), ...; excludes the entity axis.
    tensor_rank : int
        Physical tensor rank, ``len(component_shape)``.
    num_components : int
        Number of scalar entries in the physical tensor,
        ``prod(component_shape)``.
    dimension : PhysicalDimensions or None
        Optional physical dimension of the field.
    export : bool
        Whether the field is written to output.
    num_single_sided : int
        Number of ordinary single-sided internal faces.
    single_mask : torch.Tensor
        Boolean mask of single-sided internal faces, shape ``[F_internal]``.
    single_data : torch.Tensor
        Single-sided face values, shape ``[F_single, *component_shape]``.
    immersed_upper : torch.Tensor
        Upper-side values, shape ``[F_immersed, *component_shape]``.
    immersed_lower : torch.Tensor
        Lower-side values, shape ``[F_immersed, *component_shape]``.
    domain_bnd_data : torch.Tensor
        Domain-boundary face values, shape ``[F_bnd, *component_shape]``.

    Notes
    -----
    Packed layout along axis 0 is
    ``[single | domain_bnd | immersed_upper | immersed_lower]``.
    Immersed blocks are present only on ``AxisProjectedGrid``.
    """

    def __init__(
        self,
        grid: GridBase,
        name: str,
        role: FieldRole,
        component_shape: tuple[int, ...],
        dimension: DimensionLike = None,
        export: bool = True,
    ):
        self._grid = grid
        self._name = name
        self._role = role
        self._component_shape = validate_component_shape(component_shape)
        self._dimension = resolve_field_dimension(name, explicit=dimension)
        self._export = export

        device = grid.device
        dtype = grid.dtype

        # Domain-boundary data
        domain_bnd_shape = (grid.num_domain_bnd_faces, *self.component_shape)
        self._domain_bnd_data = torch.zeros(
            domain_bnd_shape, dtype=dtype, device=device
        )

        self._num_single_sided = grid.num_internal_faces
        self._single_mask = torch.ones(
            grid.num_internal_faces, dtype=torch.bool, device=device
        )
        # Immersed-boundary data (double-sided)
        if isinstance(grid, AxisProjectedGrid):
            self._single_mask[grid.ap_is_immersed_faces] = False
            immersed_bnd_shape = (
                grid.num_immersed_faces,
                *self.component_shape,
            )
            self._immersed_upper = torch.zeros(
                immersed_bnd_shape, dtype=dtype, device=device
            )
            self._immersed_lower = torch.zeros(
                immersed_bnd_shape, dtype=dtype, device=device
            )
            self._num_single_sided -= grid.num_immersed_faces

        # Internal-face data (single-sided subset only)
        shape = (self._num_single_sided, *self.component_shape)
        self._single_data = torch.zeros(shape, dtype=dtype, device=device)

        # Self-register in the grid field registry.
        grid.register_facefield(self)

        self.update_history(reset=True)

    def update_history(self, *, reset: bool = False) -> None:
        """
        Store the current single-sided face values as the old time level.

        An independent copy is always kept so that ``fvc.ddt_corr`` sees the
        previous time-step flux even when the current values are updated in
        place. The flux field is created as ``FieldRole.LOCAL`` by several
        producers (potential-flow initialisation, diagnostics), so the role
        is not used to decide whether to copy.
        """
        self._older_single_data = None if reset else self._old_single_data
        self._previous_dt = None if reset else self.grid.dt
        self._old_single_data = self._single_data.clone()

    @property
    def old_single_data(
        self,
    ) -> Float[torch.Tensor, " F_single *component_shape"]:
        """Previous single-sided values, ``[F_single, *component_shape]``."""
        return self._old_single_data

    @property
    def older_single_data(self) -> torch.Tensor | None:
        """Second old single-sided flux, or None after history reset."""
        return self._older_single_data

    @property
    def previous_dt(self) -> float | None:
        """Time interval between the two stored old flux levels."""
        return self._previous_dt

    def state_token(self) -> tuple[int, ...]:
        """
        Fingerprint of all face blocks used for cache invalidation.

        See :meth:`CellField.state_token`.
        """
        token: tuple[int, ...] = ()
        for block in self._pack_blocks():
            token += _tensor_token(block)
        return token

    def sync_to_grid_topology(self, *, topology_changed: bool = False) -> None:
        """
        Resize face buffers to match the current grid IBM / topology.

        Parameters
        ----------
        topology_changed : bool, default False
            If True, internal-face and domain-boundary counts may have
            changed and all buffers are reallocated. If False, topology is
            assumed fixed and single-sided values are preserved on faces
            that remain non-immersed.
        """
        grid = self.grid
        n_internal = grid.num_internal_faces
        n_bnd = grid.num_domain_bnd_faces
        component_shape = self.component_shape
        device = grid.device
        dtype = grid.dtype

        new_mask = torch.ones(n_internal, dtype=torch.bool, device=device)
        if isinstance(grid, AxisProjectedGrid):
            new_mask[grid.ap_is_immersed_faces] = False
            immersed_shape = (grid.num_immersed_faces, *component_shape)
            self._immersed_upper = torch.zeros(
                immersed_shape, dtype=dtype, device=device
            )
            self._immersed_lower = torch.zeros(
                immersed_shape, dtype=dtype, device=device
            )

        if topology_changed or self._single_mask.shape[0] != n_internal:
            self._single_mask = new_mask
            self._num_single_sided = int(new_mask.sum().item())
            self._single_data = torch.zeros(
                (self._num_single_sided, *component_shape),
                dtype=dtype,
                device=device,
            )
            self._domain_bnd_data = torch.zeros(
                (n_bnd, *component_shape), dtype=dtype, device=device
            )
        else:
            full = torch.zeros(
                (n_internal, *component_shape), dtype=dtype, device=device
            )
            full[self._single_mask] = self._single_data
            self._single_mask = new_mask
            self._single_data = full[new_mask]
            self._num_single_sided = int(new_mask.sum().item())
        # The previous time level is undefined on the new face set.
        self.update_history(reset=True)

    # ================================
    # Grid Accessors
    # ================================
    @property
    def grid(self) -> GridBase:
        """Computational grid that owns this field."""
        return self._grid

    # ================================
    # Metadata Accessors
    # ================================
    @property
    def name(self) -> str:
        """Field name, optionally with a phase suffix."""
        return self._name

    @property
    def role(self) -> FieldRole:
        """Temporal role of the field."""
        return self._role

    @property
    def dimension(self) -> PhysicalDimensions | None:
        """Optional physical dimension of the field."""
        return self._dimension

    @property
    def export(self) -> bool:
        """Whether the field is written to output."""
        return self._export

    # ================================
    # Data Accessors
    # ================================
    @property
    def num_single_sided(self) -> int:
        """Number of ordinary single-sided internal faces."""
        return self._num_single_sided

    @property
    def single_mask(self) -> Bool[torch.Tensor, " F_internal "]:
        """Boolean mask of single-sided internal faces ``[F_internal]``."""
        return self._single_mask

    @property
    def single_data(self) -> Float[torch.Tensor, " F_single *component_shape"]:
        """Single-sided face values ``[F_single, *component_shape]``."""
        return self._single_data

    @single_data.setter
    def single_data(
        self, value: Float[torch.Tensor, " F_single *component_shape"]
    ):
        require_shape(
            value,
            (self.num_single_sided, *self.component_shape),
            f"{self.name}.single_data",
        )
        self._single_data = value

    @property
    def immersed_upper(
        self,
    ) -> Float[torch.Tensor, " F_immersed *component_shape"]:
        """Upper-side values, shape ``[F_immersed, *component_shape]``."""
        if not isinstance(self.grid, AxisProjectedGrid):
            raise ValueError(
                "Immersed upper data is not available for this grid."
            )
        return self._immersed_upper

    @immersed_upper.setter
    def immersed_upper(
        self, value: Float[torch.Tensor, " F_immersed *component_shape"]
    ):
        if not isinstance(self.grid, AxisProjectedGrid):
            raise ValueError(
                "Immersed upper data is not available for this grid."
            )
        require_shape(
            value,
            (self.grid.num_immersed_faces, *self.component_shape),
            f"{self.name}.immersed_upper",
        )
        self._immersed_upper = value

    @property
    def immersed_lower(
        self,
    ) -> Float[torch.Tensor, " F_immersed *component_shape"]:
        """Lower-side values, shape ``[F_immersed, *component_shape]``."""
        if not isinstance(self.grid, AxisProjectedGrid):
            raise ValueError(
                "Immersed lower data is not available for this grid."
            )
        return self._immersed_lower

    @immersed_lower.setter
    def immersed_lower(
        self, value: Float[torch.Tensor, " F_immersed *component_shape"]
    ):
        if not isinstance(self.grid, AxisProjectedGrid):
            raise ValueError(
                "Immersed lower data is not available for this grid."
            )
        require_shape(
            value,
            (self.grid.num_immersed_faces, *self.component_shape),
            f"{self.name}.immersed_lower",
        )
        self._immersed_lower = value

    @property
    def domain_bnd_data(self) -> Float[torch.Tensor, " F_bnd *component_shape"]:
        """Domain-boundary face values, shape ``[F_bnd, *component_shape]``."""
        return self._domain_bnd_data

    @domain_bnd_data.setter
    def domain_bnd_data(
        self, value: Float[torch.Tensor, " F_bnd *component_shape"]
    ):
        require_shape(
            value,
            (self.grid.num_domain_bnd_faces, *self.component_shape),
            f"{self.name}.domain_bnd_data",
        )
        self._domain_bnd_data = value

    def packed_n_rows(self) -> int:
        """Packed layout entity count, excluding the physical tensor axes."""
        return sum(block.shape[0] for block in self._pack_blocks())

    def pack(self) -> Float[torch.Tensor, " N *component_shape"]:
        """
        Concatenate face blocks along axis 0.

        Layout is ``[single | domain_bnd | immersed_upper | immersed_lower]``.
        Immersed blocks are omitted unless the grid is an
        ``AxisProjectedGrid``.
        """
        return torch.cat(self._pack_blocks(), dim=0)

    def unpack(
        self, packed: Float[torch.Tensor, " N *component_shape"]
    ) -> None:
        """
        Write a packed tensor back into the face blocks.

        Parameters
        ----------
        packed : torch.Tensor
            Packed values with shape ``[N, *component_shape]``.

        Raises
        ------
        ValueError
            If the row count differs from ``packed_n_rows()`` or the physical
            tensor axes differ from ``component_shape``.
        """
        require_shape(
            packed,
            (self.packed_n_rows(), *self.component_shape),
            f"{self.name}.packed",
        )
        offset = 0
        for block in self._pack_blocks():
            n_block = block.shape[0]
            block[:] = packed[offset : offset + n_block]
            offset += n_block

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        """
        Move face buffers to ``device`` in place.

        Parameters
        ----------
        device : torch.device or str
            Target device.
        non_blocking : bool, default False
            Passed through to ``Tensor.to``.

        Returns
        -------
        FaceField
            This field after the buffers have been moved.
        """
        self._single_data = self._single_data.to(
            device=device, non_blocking=non_blocking
        )
        self._old_single_data = self._old_single_data.to(
            device=device, non_blocking=non_blocking
        )
        if self._older_single_data is not None:
            self._older_single_data = self._older_single_data.to(
                device=device, non_blocking=non_blocking
            )
        self._domain_bnd_data = self._domain_bnd_data.to(
            device=device, non_blocking=non_blocking
        )
        self._single_mask = self._single_mask.to(
            device=device, non_blocking=non_blocking
        )
        if isinstance(self.grid, AxisProjectedGrid):
            self._immersed_upper = self._immersed_upper.to(
                device=device, non_blocking=non_blocking
            )
            self._immersed_lower = self._immersed_lower.to(
                device=device, non_blocking=non_blocking
            )
        return self

    def _pack_blocks(self) -> list[torch.Tensor]:
        """Face-value blocks in packed-layout order."""
        blocks = [self._single_data, self._domain_bnd_data]
        if isinstance(self.grid, AxisProjectedGrid):
            blocks.extend([self._immersed_upper, self._immersed_lower])
        return blocks


def packed_face_n_rows(grid: GridBase) -> int:
    """
    Return the packed face-layout row count for ``grid``.

    Parameters
    ----------
    grid : GridBase
        Computational grid whose topology defines the packed layout.

    Returns
    -------
    int
        Number of rows ``N`` in ``[N, *component_shape]`` packed face tensors.
    """
    if isinstance(grid, AxisProjectedGrid):
        n_single = grid.num_internal_faces - grid.num_immersed_faces
        return (
            n_single + grid.num_domain_bnd_faces + 2 * grid.num_immersed_faces
        )
    return grid.num_internal_faces + grid.num_domain_bnd_faces


def get_or_create_cellfield(
    grid: GridBase,
    name: str,
    role: FieldRole,
    component_shape: tuple[int, ...],
    dimension: DimensionLike = None,
) -> CellField:
    """
    Return an existing cell field or create and register a new one.

    Parameters
    ----------
    grid : GridBase
        Computational grid that owns the field registry.
    name : str
        Field name, optionally with a phase suffix.
    role : FieldRole
        Temporal role required for the field.
    component_shape : tuple[int, ...]
        Physical tensor axes; () for a scalar.
    dimension : PhysicalDimensions or dict[str, float or int] or None, optional
        Optional physical dimension of the field.

    Returns
    -------
    CellField
        Existing or newly created cell-centered field.

    Raises
    ------
    AssertionError
        If an existing field has a mismatched ``role``.
    ValueError
        If the physical tensor shape does not match.
    DimensionMismatchError
        If an existing field has an incompatible dimension.
    """
    resolved = resolve_field_dimension(name, explicit=dimension)
    field = grid.get_cellfield(name)
    if field is None:
        field = CellField(
            grid=grid,
            name=name,
            role=role,
            component_shape=component_shape,
            dimension=dimension,
        )
    assert field.role == role
    if field.component_shape != component_shape:
        raise ValueError(
            f"Field {name!r}: component_shape {field.component_shape}, "
            f"expected {component_shape}"
        )
    assert_compatible(field.dimension, resolved, f"cell field {name!r}")
    return field


def get_or_create_facefield(
    grid: GridBase,
    name: str,
    role: FieldRole,
    component_shape: tuple[int, ...],
    dimension: DimensionLike = None,
    export: bool = True,
) -> FaceField:
    """
    Return an existing face field or create and register a new one.

    Parameters
    ----------
    grid : GridBase
        Computational grid that owns the field registry.
    name : str
        Field name, optionally with a phase suffix.
    role : FieldRole
        Temporal role required for the field.
    component_shape : tuple[int, ...]
        Physical tensor axes; () for a scalar.
    dimension : PhysicalDimensions or dict[str, float or int] or None, optional
        Optional physical dimension of the field.
    export : bool, optional
        Whether the field is written to output. Default is ``True``.

    Returns
    -------
    FaceField
        Existing or newly created face-centered field.

    Raises
    ------
    AssertionError
        If an existing field has mismatched metadata.
    DimensionMismatchError
        If an existing field has an incompatible dimension.
    """
    resolved = resolve_field_dimension(name, explicit=dimension)
    field = grid.get_facefield(name)
    if field is None:
        field = FaceField(
            grid=grid,
            name=name,
            role=role,
            component_shape=component_shape,
            dimension=dimension,
            export=export,
        )
    assert field.role == role
    if field.component_shape != component_shape:
        raise ValueError(
            f"Field {name!r}: component_shape {field.component_shape}, "
            f"expected {component_shape}"
        )
    assert_compatible(field.dimension, resolved, f"face field {name!r}")
    assert field.export == export
    return field

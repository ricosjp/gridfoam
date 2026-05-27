from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import numpy as np
import torch
from jaxtyping import Float, Int
from scipy.spatial import KDTree

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.models.turbulence.base import TurbulenceModel


@dataclass(frozen=True)
class ForceCoefficients:
    force: Float[torch.Tensor, " 3"]
    pressure_force: Float[torch.Tensor, " 3"]
    viscous_force: Float[torch.Tensor, " 3"]
    coefficient: Float[torch.Tensor, " 3"]
    pressure_coefficient: Float[torch.Tensor, " 3"]
    viscous_coefficient: Float[torch.Tensor, " 3"]
    cd: Float[torch.Tensor, ""]
    pressure_cd: Float[torch.Tensor, ""]
    viscous_cd: Float[torch.Tensor, ""]


@dataclass(frozen=True)
class SurfaceGeometry:
    centroids: Float[torch.Tensor, " F_surf 3"]
    area_vectors: Float[torch.Tensor, " F_surf 3"]
    normals: Float[torch.Tensor, " F_surf 3"]
    areas: Float[torch.Tensor, " F_surf 1"]


@dataclass(frozen=True)
class ProbeState:
    distance: Float[torch.Tensor, " F_surf 1"]
    pressure: Float[torch.Tensor, " F_surf 1"]
    velocity: Float[torch.Tensor, " F_surf 3"]
    nu_eff: Float[torch.Tensor, " F_surf 1"]


@dataclass(frozen=True)
class ForceInputs:
    drag_direction: Float[torch.Tensor, " 3"]
    coefficient_denominator: Float[torch.Tensor, ""]
    wall_velocity: Float[torch.Tensor, " 3"]


_LOCATOR_CACHE: WeakKeyDictionary[AxisProjectedGrid, KDTree] = (
    WeakKeyDictionary()
)


def compute_force_coefficients(
    grid: IGridBase,
    U: CellField,
    p: CellField,
    turbulence: TurbulenceModel,
    *,
    drag_direction: Sequence[float] = (1.0, 0.0, 0.0),
    reference_velocity: float,
    reference_area: float,
    probe_offset_factor: float = 1.0,
    wall_velocity: Sequence[float] | None = None,
) -> ForceCoefficients:
    """
    Compute force coefficients by surface-mesh integration.

    For each polygon face of ``grid.surface_mesh`` the wall traction

    .. math::

        \\mathbf{t} = -p\\,\\mathbf{n}
        + \\nu_{\\mathrm{eff}}\\,(\\partial_n \\mathbf{U})_t

    is evaluated by sampling the Eulerian fields at a fluid-side probe
    point ``x_p = c + h\\,\\mathbf{n}`` placed along the outward unit normal
    of each face, where ``h`` is the local background cell size scaled by
    ``probe_offset_factor``. The wall-normal velocity gradient is
    reconstructed via a one-sided difference between the probe and the
    wall (no-slip by default), and the tangential component of that
    gradient drives the viscous traction. Pressure is sampled at the same
    probe point under the boundary-layer assumption that
    ``\\partial p/\\partial n \\approx 0``. Total force on the body is
    obtained by summing ``\\mathbf{t}\\,dA`` over every polygon face.

    Parameters
    ----------
    grid : IGridBase
        Computational grid. Must be an :class:`AxisProjectedGrid` whose
        ``surface_mesh`` is available.
    U : CellField
        Cell-centred velocity field (3 components).
    p : CellField
        Cell-centred kinematic pressure field (1 component).
    turbulence : TurbulenceModel
        Turbulence model providing the effective kinematic viscosity.
    drag_direction : sequence of float, optional
        Vector defining the drag direction. Internally normalized.
        Defaults to the +x axis.
    reference_velocity : float
        Free-stream reference velocity ``U_inf`` (must be positive).
    reference_area : float
        Reference area ``A_ref`` (must be positive).
    probe_offset_factor : float, default=1.0
        Probe distance in units of the local background cell size. Must
        be strictly positive; values smaller than 1 may land inside a
        cut cell.
    wall_velocity : sequence of float or None, optional
        Wall velocity used as the Dirichlet value for the velocity
        gradient reconstruction (e.g. for moving bodies). ``None`` (the
        default) is treated as a no-slip wall.

    Returns
    -------
    ForceCoefficients
        Decomposition of the body force together with the normalized
        coefficients and drag-direction projections.

    Notes
    -----
    Pressure is assumed to be the kinematic pressure ``p/rho`` and the
    effective viscosity is kinematic, consistent with the rest of the
    incompressible pressure convention used by the solver.
    """
    _validate_inputs(grid, U, p, reference_velocity, reference_area)
    assert isinstance(grid, AxisProjectedGrid)

    inputs = _build_force_inputs(
        grid,
        drag_direction=drag_direction,
        reference_velocity=reference_velocity,
        reference_area=reference_area,
        wall_velocity=wall_velocity,
    )
    surface = _surface_geometry(grid)
    probe = _sample_probe_state(
        grid,
        U,
        p,
        turbulence,
        surface,
        probe_offset_factor=probe_offset_factor,
    )
    pressure_force, viscous_force = _integrate_surface_forces(
        surface, probe, inputs.wall_velocity
    )

    return _force_coefficients(
        pressure_force,
        viscous_force,
        coefficient_denominator=inputs.coefficient_denominator,
        drag_direction=inputs.drag_direction,
    )


def _validate_inputs(
    grid: IGridBase,
    U: CellField,
    p: CellField,
    reference_velocity: float,
    reference_area: float,
) -> None:
    if not isinstance(grid, AxisProjectedGrid):
        raise TypeError("force coefficients require an AxisProjectedGrid.")
    if U.grid is not grid or p.grid is not grid:
        raise ValueError("U and p must belong to the supplied grid.")
    if U.num_components != 3:
        raise ValueError("U must be a 3-component vector field.")
    if p.num_components != 1:
        raise ValueError("p must be a scalar field.")
    if reference_velocity <= 0.0:
        raise ValueError("reference_velocity must be positive.")
    if reference_area <= 0.0:
        raise ValueError("reference_area must be positive.")


def _build_force_inputs(
    grid: AxisProjectedGrid,
    *,
    drag_direction: Sequence[float],
    reference_velocity: float,
    reference_area: float,
    wall_velocity: Sequence[float] | None,
) -> ForceInputs:
    dtype = grid.dtype
    device = grid.device
    return ForceInputs(
        drag_direction=_unit_vector(drag_direction, dtype=dtype, device=device),
        coefficient_denominator=torch.tensor(
            0.5 * reference_velocity**2 * reference_area,
            dtype=dtype,
            device=device,
        ),
        wall_velocity=_wall_velocity_tensor(
            wall_velocity, dtype=dtype, device=device
        ),
    )


def _unit_vector(
    values: Sequence[float], *, dtype: torch.dtype, device: torch.device
) -> Float[torch.Tensor, " 3"]:
    direction = torch.tensor(values, dtype=dtype, device=device)
    if direction.shape != (3,):
        raise ValueError("drag_direction must contain exactly 3 values.")
    norm = torch.linalg.vector_norm(direction)
    if norm <= 0.0:
        raise ValueError("drag_direction must be non-zero.")
    return direction / norm


def _wall_velocity_tensor(
    wall_velocity: Sequence[float] | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Float[torch.Tensor, " 3"]:
    if wall_velocity is None:
        return torch.zeros(3, dtype=dtype, device=device)
    tensor = torch.tensor(wall_velocity, dtype=dtype, device=device)
    if tensor.shape != (3,):
        raise ValueError("wall_velocity must contain exactly 3 values.")
    return tensor


def _surface_geometry(
    grid: AxisProjectedGrid,
) -> SurfaceGeometry:
    """
    Return per-polygon centroids, area vectors, unit normals, and areas.

    Each row corresponds to one (general polygonal) face of the immersed
    surface mesh. The face area vector ``S = n |A|`` is computed by
    graphlow's polygon-aware geometry helpers. The vectors are then
    oriented so that the signed enclosed volume is positive, matching
    the convention "normal points from the body into the fluid".
    """
    surface = grid.surface_mesh
    centroids = surface.geometry.face_centroids().to(
        dtype=grid.dtype, device=grid.device
    )
    area_vectors = surface.geometry.face_area_vectors().to(
        dtype=grid.dtype, device=grid.device
    )
    area_vectors = _orient_area_vectors_outward(centroids, area_vectors)
    areas = torch.linalg.vector_norm(area_vectors, dim=1, keepdim=True)
    safe_areas = torch.clamp_min(areas, torch.finfo(grid.dtype).tiny)
    normals = area_vectors / safe_areas
    return SurfaceGeometry(
        centroids=centroids,
        area_vectors=area_vectors,
        normals=normals,
        areas=areas,
    )


def _orient_area_vectors_outward(
    centroids: Float[torch.Tensor, " F_surf 3"],
    area_vectors: Float[torch.Tensor, " F_surf 3"],
) -> Float[torch.Tensor, " F_surf 3"]:
    """
    Orient surface area vectors to point out of the immersed body.

    The surface traction integral uses ``F = ∫ (-p n + tau n) dA`` with
    ``n`` pointing from the body into the fluid. For a closed surface
    this corresponds to a positive signed volume
    ``V = 1/3 ∫ x · n dA``.
    """
    signed_volume = torch.sum(centroids * area_vectors) / 3.0
    if signed_volume < 0.0:
        return -area_vectors
    return area_vectors


def _sample_probe_state(
    grid: AxisProjectedGrid,
    U: CellField,
    p: CellField,
    turbulence: TurbulenceModel,
    surface: SurfaceGeometry,
    *,
    probe_offset_factor: float,
) -> ProbeState:
    if probe_offset_factor <= 0.0:
        raise ValueError("probe_offset_factor must be positive.")

    distance = _probe_distance(grid, surface.centroids, probe_offset_factor)
    probe_points = surface.centroids + distance * surface.normals
    probe_cells = _locate_cells(grid, probe_points)

    return ProbeState(
        distance=distance,
        pressure=p.data[probe_cells],
        velocity=U.data[probe_cells],
        nu_eff=turbulence.nu_eff()[probe_cells],
    )


def _probe_distance(
    grid: AxisProjectedGrid,
    centroids: Float[torch.Tensor, " F_surf 3"],
    probe_offset_factor: float,
) -> Float[torch.Tensor, " F_surf 1"]:
    surface_cells = _locate_cells(grid, centroids)
    local_cell_sizes = grid.cell_sizes[surface_cells]
    return probe_offset_factor * local_cell_sizes.amax(dim=1, keepdim=True)


def _integrate_surface_forces(
    surface: SurfaceGeometry,
    probe: ProbeState,
    wall_velocity: Float[torch.Tensor, " 3"],
) -> tuple[Float[torch.Tensor, " 3"], Float[torch.Tensor, " 3"]]:
    dUdn = (probe.velocity - wall_velocity) / probe.distance
    dUdn_tangential = _tangential_part(dUdn, surface.normals)

    # Force on the body from the fluid: F = ∮ (-p n + ν_eff (∂U/∂n)_t) dA.
    pressure_force = -(probe.pressure * surface.area_vectors).sum(dim=0)
    viscous_force = (probe.nu_eff * surface.areas * dUdn_tangential).sum(dim=0)
    return pressure_force, viscous_force


def _tangential_part(
    vector: Float[torch.Tensor, " F_surf 3"],
    normals: Float[torch.Tensor, " F_surf 3"],
) -> Float[torch.Tensor, " F_surf 3"]:
    normal_part = torch.sum(vector * normals, dim=1, keepdim=True) * normals
    return vector - normal_part


def _force_coefficients(
    pressure_force: Float[torch.Tensor, " 3"],
    viscous_force: Float[torch.Tensor, " 3"],
    *,
    coefficient_denominator: Float[torch.Tensor, ""],
    drag_direction: Float[torch.Tensor, " 3"],
) -> ForceCoefficients:
    force = pressure_force + viscous_force
    coefficient = force / coefficient_denominator
    pressure_coefficient = pressure_force / coefficient_denominator
    viscous_coefficient = viscous_force / coefficient_denominator

    return ForceCoefficients(
        force=force,
        pressure_force=pressure_force,
        viscous_force=viscous_force,
        coefficient=coefficient,
        pressure_coefficient=pressure_coefficient,
        viscous_coefficient=viscous_coefficient,
        cd=torch.dot(coefficient, drag_direction),
        pressure_cd=torch.dot(pressure_coefficient, drag_direction),
        viscous_cd=torch.dot(viscous_coefficient, drag_direction),
    )


def _locate_cells(
    grid: AxisProjectedGrid,
    points: Float[torch.Tensor, " N 3"],
    *,
    num_candidates: int = 1,
    atol: float = 1e-9,
) -> Int[torch.Tensor, " N"]:
    """
    Locate the background cell that owns each query point.

    Uses a cell-center KD-tree to fetch nearby cell candidates, then
    picks the first candidate whose axis-aligned bounding box contains
    the point. The default uses only the nearest cell center because the
    immersed-boundary neighbourhood is expected to be locally refined.
    Larger ``num_candidates`` values can be used when ownership near a
    strong AMR transition must be recovered more conservatively. If no
    candidate contains the point, the nearest center is returned as a
    fallback.
    """
    tree = _get_cell_locator(grid)
    points_np = points.detach().cpu().numpy()
    centers_np = grid.cell_centers.detach().cpu().numpy()
    sizes_np = grid.cell_sizes.detach().cpu().numpy()

    k_eff = min(num_candidates, centers_np.shape[0])
    _, raw_idxs = tree.query(points_np, k=k_eff)
    # scipy returns squeezed shapes when k=1 or when there is a single
    # query point; normalize to (n_queries, k_eff) for indexing.
    idxs = np.atleast_1d(np.asarray(raw_idxs, dtype=np.intp))
    if idxs.ndim == 1:
        idxs = idxs[:, np.newaxis]

    cand_centers = centers_np[idxs]
    cand_sizes = sizes_np[idxs]
    half = cand_sizes / 2.0
    diff = np.abs(points_np[:, None, :] - cand_centers)
    inside = np.all(diff <= half + atol, axis=-1)

    has_inside = np.any(inside, axis=1)
    first_inside = inside.argmax(axis=1)
    row = np.arange(int(idxs.shape[0]), dtype=np.intp)
    chosen = np.where(has_inside, idxs[row, first_inside], idxs[:, 0])

    return torch.from_numpy(chosen).to(device=points.device, dtype=torch.long)


def _get_cell_locator(grid: AxisProjectedGrid) -> KDTree:
    cached = _LOCATOR_CACHE.get(grid)
    if cached is not None:
        return cached
    centers_np = grid.cell_centers.detach().cpu().numpy()
    tree = KDTree(centers_np)
    _LOCATOR_CACHE[grid] = tree
    return tree

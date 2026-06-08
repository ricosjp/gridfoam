"""
End-to-end checks for :mod:`gridfoam.post.forces`.

These tests construct controlled AP-IBM scenarios where the integrated
force and moment can be computed analytically (or asymptotically) and
compare against :class:`ForceEvaluator`.

The cases are:

1. ``OrthonormalCoord`` constructs a right-handed unit basis (drag/lift
   and drag/pitch modes).
2. ``_surface_Sf`` reconstructs the true surface area vector from the
   AP-projected area vector while preserving the AP-side orientation.
3. A uniform kinematic pressure on a closed body produces exactly zero
   net force and moment.
4. A pressure jump across a thin sheet converges to the analytical lift
   force as the AP grid is refined.
5. A low-Reynolds SIMPLE simulation of a cube in cross-flow produces a
   non-trivial, well-conditioned drag coefficient. Marked ``slow``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from gridfoam.algorithms.simple import SIMPLE
from gridfoam.boundaries.factory import apply_boundary_condition_configs
from gridfoam.core.field import CellField
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import (
    DragLiftCoord,
    DragPitchCoord,
    GridfoamConfig,
)
from gridfoam.meta.enums import (
    FieldRole,
    ForceCoordMode,
)
from gridfoam.models.turbulence.laminar import Laminar
from gridfoam.post.forces import ForceEvaluator, OrthonormalCoord

# ---------------------------------------------------------------------------
# 1. Local coordinate construction
# ---------------------------------------------------------------------------


def test_orthonormal_coord_drag_lift_is_right_handed() -> None:
    coord = OrthonormalCoord.from_local_coord(
        DragLiftCoord(
            mode=ForceCoordMode.DRAG_LIFT,
            drag_dir=[2.0, 0.0, 0.0],
            lift_dir=[1.0, 0.0, 3.0],
            center_of_rotation=[0.0, 0.0, 0.0],
        )
    )
    e1, e2, e3 = coord.e1, coord.e2, coord.e3
    assert math.isclose(torch.linalg.norm(e1).item(), 1.0, abs_tol=1e-12)
    assert math.isclose(torch.linalg.norm(e2).item(), 1.0, abs_tol=1e-12)
    assert math.isclose(torch.linalg.norm(e3).item(), 1.0, abs_tol=1e-12)
    assert abs(torch.dot(e1, e2).item()) < 1e-12
    assert abs(torch.dot(e2, e3).item()) < 1e-12
    assert abs(torch.dot(e3, e1).item()) < 1e-12
    assert torch.allclose(torch.linalg.cross(e1, e2), e3, atol=1e-12)
    assert torch.allclose(
        e1, torch.tensor([1.0, 0.0, 0.0], dtype=e1.dtype), atol=1e-12
    )
    assert torch.allclose(
        e3, torch.tensor([0.0, 0.0, 1.0], dtype=e3.dtype), atol=1e-12
    )


def test_orthonormal_coord_drag_pitch_is_right_handed() -> None:
    coord = OrthonormalCoord.from_local_coord(
        DragPitchCoord(
            mode=ForceCoordMode.DRAG_PITCH,
            drag_dir=[1.0, 0.0, 0.0],
            pitch_axis=[0.5, 1.0, 0.0],
            center_of_rotation=[0.0, 0.0, 0.0],
        )
    )
    e1, e2, e3 = coord.e1, coord.e2, coord.e3
    assert math.isclose(torch.linalg.norm(e1).item(), 1.0, abs_tol=1e-12)
    assert math.isclose(torch.linalg.norm(e2).item(), 1.0, abs_tol=1e-12)
    assert math.isclose(torch.linalg.norm(e3).item(), 1.0, abs_tol=1e-12)
    assert abs(torch.dot(e1, e2).item()) < 1e-12
    assert abs(torch.dot(e2, e3).item()) < 1e-12
    assert abs(torch.dot(e3, e1).item()) < 1e-12
    assert torch.allclose(torch.linalg.cross(e1, e2), e3, atol=1e-12)
    assert torch.allclose(
        e2, torch.tensor([0.0, 1.0, 0.0], dtype=e2.dtype), atol=1e-12
    )


# ---------------------------------------------------------------------------
# 3. Uniform pressure on a closed body gives exactly zero force / moment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", ["cube", "flat_plate"])
def test_uniform_pressure_gives_zero_force_on_closed_body(
    tmp_path: Path,
    body: str,
    make_cube_stl: Callable[..., None],
    make_thin_plate_stl: Callable[..., None],
    build_force_config: Callable[..., GridfoamConfig],
    add_dirichlet_bc: Callable[..., None],
    add_neumann_bc: Callable[..., None],
) -> None:
    stl = tmp_path / f"body_{body}.stl"
    if body == "cube":
        make_cube_stl(center=(0.5, 0.5, 0.5), side=0.2, out_path=stl)
    else:
        make_thin_plate_stl(
            center=(0.5, 0.5, 0.5),
            length_x=0.4,
            length_y=0.4,
            out_path=stl,
            tilt_deg=0.0,
        )

    config = build_force_config(
        stl_path=stl,
        root_resolution=[16, 16, 16],
        output_dir=tmp_path / "out",
    )
    grid = create_grid(config)
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    p = CellField(grid, "p", role=FieldRole.LOCAL, num_components=1)
    p.data.fill_(2.5)
    add_dirichlet_bc(U, "_default", [0.0, 0.0, 0.0])
    add_neumann_bc(p, "_default", [0.0])

    fc = config.simulator.forceCoeff
    assert fc is not None
    evaluator = ForceEvaluator(fc)
    co = evaluator.evaluate(
        grid,
        time=0.0,
        p=p,
        U=U,
        turbulence=Laminar(grid=grid, nu=0.1),
    )

    tol = 1e-10
    assert abs(co.Cd.item()) < tol
    assert abs(co.Cs.item()) < tol
    assert abs(co.Cl.item()) < tol
    assert abs(co.CmRoll.item()) < tol
    assert abs(co.CmPitch.item()) < tol
    assert abs(co.CmYaw.item()) < tol


# ---------------------------------------------------------------------------
# 4. Thin sheet with a pressure jump converges to the analytical lift
# ---------------------------------------------------------------------------


def _run_pressure_jump(
    *,
    tmp_path: Path,
    root_resolution: list[int],
    p_above: float,
    p_below: float,
    make_thin_plate_stl: Callable[..., None],
    build_force_config: Callable[..., GridfoamConfig],
    add_dirichlet_bc: Callable[..., None],
    add_neumann_bc: Callable[..., None],
) -> tuple[float, float]:
    """
    Run the thin-sheet pressure-jump case and return ``(Cl, Cl_expected)``.

    AP-IBM's two-sided integration treats the sheet as a truly thin
    surface: the upper-side integration uses p just above the sheet and
    the lower-side integration uses p just below. The expected lift on
    the body in +z is therefore rho * (p_below - p_above) * area.
    """
    stl = tmp_path / "thin_sheet.stl"
    half = 0.2
    make_thin_plate_stl(
        center=(0.5, 0.5, 0.5),
        length_x=2 * half,
        length_y=2 * half,
        out_path=stl,
        tilt_deg=0.0,
        thickness=2e-3,
    )
    config = build_force_config(
        stl_path=stl,
        root_resolution=root_resolution,
        output_dir=tmp_path / "out",
    )
    grid = create_grid(config)
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    p = CellField(grid, "p", role=FieldRole.LOCAL, num_components=1)
    z = grid.cell_centers[:, 2]
    p.data = (
        torch.where(z > 0.5, p_above, p_below).unsqueeze(-1).to(p.data.dtype)
    )
    add_dirichlet_bc(U, "_default", [0.0, 0.0, 0.0])
    add_neumann_bc(p, "_default", [0.0])

    fc = config.simulator.forceCoeff
    assert fc is not None
    evaluator = ForceEvaluator(fc)
    co = evaluator.evaluate(
        grid,
        time=0.0,
        p=p,
        U=U,
        turbulence=Laminar(grid=grid, nu=0.0),
    )
    area = (2.0 * half) ** 2
    F_z_expected = (p_below - p_above) * area
    Cl_expected = F_z_expected / 0.5
    return co.Cl.item(), Cl_expected


def test_thin_sheet_pressure_jump_converges_with_refinement(
    tmp_path: Path,
    make_thin_plate_stl: Callable[..., None],
    build_force_config: Callable[..., GridfoamConfig],
    add_dirichlet_bc: Callable[..., None],
    add_neumann_bc: Callable[..., None],
) -> None:
    p_above, p_below = 1.0, 2.0
    errors: list[float] = []
    for res in ([16, 16, 16], [32, 32, 32], [64, 64, 64]):
        sub = tmp_path / f"res_{res[0]}"
        sub.mkdir()
        cl, cl_expected = _run_pressure_jump(
            tmp_path=sub,
            root_resolution=res,
            p_above=p_above,
            p_below=p_below,
            make_thin_plate_stl=make_thin_plate_stl,
            build_force_config=build_force_config,
            add_dirichlet_bc=add_dirichlet_bc,
            add_neumann_bc=add_neumann_bc,
        )
        rel = abs(cl - cl_expected) / abs(cl_expected)
        errors.append(rel)
    # Finest grid must be within 10 % of analytical.
    assert errors[-1] < 0.10, f"too large error on fine grid: {errors}"
    # Refinement must not blow up the error.
    assert errors[-1] <= errors[0] + 1e-12, f"refinement did not help: {errors}"


# ---------------------------------------------------------------------------
# 5. SIMPLE simulation of a cube in cross-flow (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cube_low_re_drag_with_simple(
    tmp_path: Path,
    make_cube_stl: Callable[..., None],
    build_cube_simple_config: Callable[..., GridfoamConfig],
) -> None:
    """
    Run a low-Reynolds (Re ~ 12) cube simulation and check Cd is sane.

    The exact Cd at this Reynolds number is not known analytically, but
    Stokes-flow estimates and prior literature put Cd in O(1) for a
    cube. We only check that the value is positive, finite and stays
    within a reasonable envelope, which is enough to detect grossly
    incorrect force integration.
    """
    stl = tmp_path / "cube_big.stl"
    make_cube_stl(center=(0.5, 0.0, 1.0), side=0.6, out_path=stl)

    config = build_cube_simple_config(
        stl_path=stl,
        output_dir=tmp_path / "out",
        root_resolution=[40, 16, 16],
        domain_lower=(-2.0, -1.0, 0.0),
        domain_upper=(+3.0, +1.0, 2.0),
        cor=(0.5, 0.0, 1.0),
        A_ref=0.36,
        L_ref=0.6,
        end_time=20.0,
    )
    grid = create_grid(config)
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    p = CellField(grid, "p", role=FieldRole.LOCAL, num_components=1)
    bcs = grid.sim_config.boundaryConditions
    assert bcs is not None
    apply_boundary_condition_configs(U, bcs["U"])
    apply_boundary_condition_configs(p, bcs["p"])

    turb = Laminar(grid=grid, nu=0.05)
    fc = grid.sim_config.forceCoeff
    assert fc is not None
    evaluator = ForceEvaluator(fc)
    algo = SIMPLE(grid=grid, U=U, p=p, turbulence=turb)
    n_steps = int(
        grid.sim_config.control.endTime / grid.sim_config.control.deltaT
    )
    for _ in range(n_steps):
        algo.step()
    co = evaluator.evaluate(
        grid, time=float(n_steps), p=p, U=U, turbulence=turb
    )

    cd = co.Cd.item()
    cl = co.Cl.item()
    assert math.isfinite(cd) and math.isfinite(cl)
    assert 0.1 < cd < 20.0, f"Cd={cd} outside the sane envelope"
    # Channel flow with slip top/bottom is symmetric -> Cl ~ 0.
    assert abs(cl) < 1.0, f"Cl={cl} too large for a symmetric setup"

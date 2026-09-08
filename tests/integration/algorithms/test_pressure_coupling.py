"""
Pressure correction conserves cell flux and honors solver scheduling.

Continuity
    SIMPLE, SIMPLEC, PISO, and PIMPLE preserve inlet/wall fluxes and cell
    continuity. Fixed pressure bypasses flux balancing.

Solver control
    Residuals are measured before solving. PISO/PIMPLE use the final solver
    on the last pressure pass; SIMPLE uses its regular solver throughout.

Coefficients and history
    SIMPLEC uses the off-diagonal H1 contribution and bounds its coefficient.
    Consistent old flux needs no ddt correction; mismatches stay bounded.
"""

from __future__ import annotations

import pathlib
from unittest.mock import Mock

import pytest
import torch
from tests.helpers import channel_flow_config

from gridfoam.algorithms.pimple import PIMPLE
from gridfoam.algorithms.piso import PISO
from gridfoam.algorithms.simple import SIMPLE
from gridfoam.algorithms.utils.pressure_correction import (
    simplec_rAtU,
    solve_pressure_poisson,
)
from gridfoam.algorithms.utils.residual import field_initial_residual
from gridfoam.core.field import CellField, FaceField, FieldRole
from gridfoam.core.grid.base import GridBase
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvc, fvm
from gridfoam.fv.adjust_phi import adjust_phi
from gridfoam.fv.flux import compute_phi_hbya
from gridfoam.meta.config import (
    PIMPLEAlgorithm,
    PISOAlgorithm,
    RelaxationFactorsConfig,
    SIMPLEAlgorithm,
)
from gridfoam.meta.enums import (
    AlgorithmType,
    BoundaryConditionType,
    DomainBoundaryPatch,
)
from gridfoam.solvers.factory import create_solver


def _simple(consistent: bool = False) -> SIMPLEAlgorithm:
    return SIMPLEAlgorithm(
        type=AlgorithmType.SIMPLE,
        consistent=consistent,
        relaxationFactors=RelaxationFactorsConfig(
            equations={"U": 0.9 if consistent else 0.7, "p": 0.3},
        ),
        pRefCell=0,
        pRefValue=0.0,
    )


# Linear solvers run to 1e-12; the cell divergence is that residual per
# unit volume, so allow a few orders of magnitude on the smallest cells.
_DIV_TOL = 1e-8


def _assert_divergence_free(phi: FaceField, tol: float = _DIV_TOL) -> None:
    div_phi = fvc.div(phi).data
    assert float(div_phi.abs().max()) < tol


def _assert_boundary_flux_matches_velocity_bcs(
    grid: GridBase, phi: FaceField
) -> None:
    inlet = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_MINUS)
    walls = grid.get_domain_bnd_mask(
        DomainBoundaryPatch.Y_MINUS
    ) | grid.get_domain_bnd_mask(DomainBoundaryPatch.Y_PLUS)
    empty = grid.get_domain_bnd_mask(
        DomainBoundaryPatch.Z_MINUS
    ) | grid.get_domain_bnd_mask(DomainBoundaryPatch.Z_PLUS)
    # Dirichlet inlet U = (1, 0, 0) against the outward normal -x.
    expected_inlet = -torch.linalg.vector_norm(
        grid.domain_bnd_Sf[inlet], dim=1, keepdim=False
    )
    torch.testing.assert_close(
        phi.domain_bnd_data[inlet], expected_inlet, atol=1e-12, rtol=0.0
    )
    # Walls (slip or no-slip) and empty patches are impermeable.
    assert float(phi.domain_bnd_data[walls].abs().max()) < 1e-12
    assert float(phi.domain_bnd_data[empty].abs().max()) < 1e-12


@pytest.mark.parametrize(
    "walls",
    [BoundaryConditionType.SLIP, BoundaryConditionType.DIRICHLET],
)
@pytest.mark.parametrize(
    "p_outlet",
    [BoundaryConditionType.DIRICHLET, BoundaryConditionType.NEUMANN],
)
def test_simple_step_flux_is_divergence_free_in_every_cell(
    tmp_path: pathlib.Path,
    p_outlet: BoundaryConditionType,
    walls: BoundaryConditionType,
) -> None:
    """
    SIMPLE conserves flux in every cell for fixed and referenced pressure.
    """
    grid = create_grid(
        channel_flow_config(tmp_path, _simple(), p_outlet=p_outlet, walls=walls)
    )
    algo = SIMPLE(grid)
    for _ in range(5):
        algo.step()

    _assert_divergence_free(algo.phi)
    _assert_boundary_flux_matches_velocity_bcs(grid, algo.phi)
    if p_outlet == BoundaryConditionType.NEUMANN:
        assert abs(float(algo.phi.domain_bnd_data.sum())) < 1e-10
    if walls == BoundaryConditionType.DIRICHLET:
        # Developing Poiseuille flow accelerates the core.
        assert float(algo.U.data[:, 0].max()) > 1.05


def test_simplec_step_flux_is_divergence_free(tmp_path: pathlib.Path) -> None:
    """
    SIMPLEC changes rAtU while preserving continuity and prescribed face
    fluxes.
    """
    grid = create_grid(
        channel_flow_config(
            tmp_path,
            _simple(consistent=True),
            walls=BoundaryConditionType.DIRICHLET,
        )
    )
    algo = SIMPLE(grid)
    for _ in range(3):
        algo.step()

    _assert_divergence_free(algo.phi)
    _assert_boundary_flux_matches_velocity_bcs(grid, algo.phi)
    # rAtU differs from rAU when SIMPLEC is active.
    assert not torch.allclose(algo.rAtU.data, algo.rAU.data)


def test_piso_step_flux_is_divergence_free(tmp_path: pathlib.Path) -> None:
    """PISO correctors preserve cell continuity and inlet/wall fluxes."""
    algorithm = PISOAlgorithm(
        type=AlgorithmType.PISO, nCorrectors=2, pRefCell=0, pRefValue=0.0
    )
    grid = create_grid(
        channel_flow_config(
            tmp_path, algorithm, walls=BoundaryConditionType.DIRICHLET
        )
    )
    algo = PISO(grid)
    for _ in range(3):
        algo.step()

    _assert_divergence_free(algo.phi)
    _assert_boundary_flux_matches_velocity_bcs(grid, algo.phi)


@pytest.mark.parametrize("consistent", [False, True])
def test_pimple_step_flux_is_divergence_free(
    tmp_path: pathlib.Path, consistent: bool
) -> None:
    """
    Both PIMPLE consistency modes preserve cell continuity and boundary
    fluxes.
    """
    algorithm = PIMPLEAlgorithm(
        type=AlgorithmType.PIMPLE,
        nCorrectors=2,
        nOuterCorrectors=2,
        consistent=consistent,
        pRefCell=0,
        pRefValue=0.0,
    )
    grid = create_grid(
        channel_flow_config(
            tmp_path, algorithm, walls=BoundaryConditionType.DIRICHLET
        )
    )
    algo = PIMPLE(grid)
    for _ in range(3):
        algo.step()

    _assert_divergence_free(algo.phi)
    _assert_boundary_flux_matches_velocity_bcs(grid, algo.phi)


def test_constrain_hbya_imposes_velocity_flux_on_fixed_patches(
    tmp_path: pathlib.Path,
) -> None:
    """
    Fixed patches use velocity flux; the adjustable outlet extrapolates
    HbyA.
    """
    grid = create_grid(channel_flow_config(tmp_path, _simple()))
    algo = SIMPLE(grid)
    HbyA = algo.HbyA
    HbyA.data = torch.rand_like(HbyA.data) + 0.5

    compute_phi_hbya(algo.phi_hbya, HbyA, algo.U)

    inlet = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_MINUS)
    outlet = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_PLUS)
    walls = grid.get_domain_bnd_mask(
        DomainBoundaryPatch.Y_MINUS
    ) | grid.get_domain_bnd_mask(DomainBoundaryPatch.Y_PLUS)

    expected_inlet = -torch.linalg.vector_norm(
        grid.domain_bnd_Sf[inlet], dim=1, keepdim=False
    )
    torch.testing.assert_close(
        algo.phi_hbya.domain_bnd_data[inlet], expected_inlet
    )
    assert float(algo.phi_hbya.domain_bnd_data[walls].abs().max()) < 1e-12
    expected_outlet = torch.sum(
        HbyA.data[grid.domain_bnd_owner[outlet]] * grid.domain_bnd_Sf[outlet],
        dim=1,
        keepdim=False,
    )
    torch.testing.assert_close(
        algo.phi_hbya.domain_bnd_data[outlet], expected_outlet
    )


def test_adjust_phi_is_noop_when_pressure_level_is_fixed(
    tmp_path: pathlib.Path,
) -> None:
    """
    Fixed pressure leaves flux unchanged; omitting pressure enforces
    balance.
    """
    grid = create_grid(channel_flow_config(tmp_path, _simple()))
    algo = SIMPLE(grid)
    algo.phi_hbya.domain_bnd_data[:] = 0.0
    inlet = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_MINUS)
    outlet = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_PLUS)
    algo.phi_hbya.domain_bnd_data[inlet] = -1.0
    algo.phi_hbya.domain_bnd_data[outlet] = 0.3
    before = algo.phi_hbya.domain_bnd_data.clone()

    assert adjust_phi(algo.phi_hbya, algo.U, algo.p) is False
    torch.testing.assert_close(algo.phi_hbya.domain_bnd_data, before)
    # Without the pressure argument the balance is enforced.
    assert adjust_phi(algo.phi_hbya, algo.U) is True
    assert abs(float(algo.phi_hbya.domain_bnd_data.sum())) < 1e-12


def test_pressure_initial_residual_is_measured_before_solve(
    tmp_path: pathlib.Path,
) -> None:
    """residualControl must see the residual of the *unsolved* system."""
    grid = create_grid(channel_flow_config(tmp_path, _simple()))
    algo = SIMPLE(grid)
    p = algo.p
    rAtU = algo.rAtU
    rAtU.data = torch.ones_like(rAtU.data)
    p.data = torch.zeros_like(p.data)
    div_source = torch.sin(grid.cell_centers[:, 0] * 3.0)

    p_eqn = -fvm.laplacian(rAtU.data, p)
    p_eqn.source = p_eqn.source - div_source * grid.cell_volumes
    expected = field_initial_residual(p_eqn, p)

    result = solve_pressure_poisson(
        p,
        rAtU,
        div_source,
        algo.solvers["p"],
        n_non_orthogonal_correctors=1,
        p_needs_ref=False,
    )

    assert result.initial_residual == pytest.approx(expected)
    assert result.initial_residual > 0.1
    assert field_initial_residual(result.matrix, p) < 1e-8


@pytest.mark.parametrize("pimple", [False, True])
def test_final_pressure_solver_only_on_last_nonorthogonal_pass(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, pimple: bool
) -> None:
    """
    Each PISO/PIMPLE pressure-corrector sequence uses pFinal only on its
    last solve.
    """
    algorithm = (
        PIMPLEAlgorithm(
            type=AlgorithmType.PIMPLE,
            nOuterCorrectors=2,
            nCorrectors=2,
            nNonOrthogonalCorrectors=2,
        )
        if pimple
        else PISOAlgorithm(
            type=AlgorithmType.PISO,
            nCorrectors=2,
            nNonOrthogonalCorrectors=2,
        )
    )
    config = channel_flow_config(tmp_path, algorithm)
    grid = create_grid(config)
    algo = PIMPLE(grid) if pimple else PISO(grid)
    regular = algo.solvers["p"]
    final = create_solver(config.simulator.fvSolution.solvers["p"])
    algo.solvers["pFinal"] = final
    calls = Mock()
    normal_spy = Mock(wraps=regular.solve)
    final_spy = Mock(wraps=final.solve)
    calls.attach_mock(normal_spy, "regular")
    calls.attach_mock(final_spy, "final")
    monkeypatch.setattr(regular, "solve", normal_spy)
    monkeypatch.setattr(final, "solve", final_spy)

    algo.step()

    expected = ["regular"] * 5 + ["final"]
    assert [call[0] for call in calls.mock_calls] == expected * (
        2 if pimple else 1
    )


def test_simple_pressure_solver_never_uses_pfinal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIMPLE uses the regular pressure solver for every nonorthogonal pass."""
    algorithm = SIMPLEAlgorithm(
        type=AlgorithmType.SIMPLE,
        nNonOrthogonalCorrectors=2,
        relaxationFactors=RelaxationFactorsConfig(
            equations={"U": 0.7, "p": 0.3},
        ),
        pRefCell=0,
        pRefValue=0.0,
    )
    config = channel_flow_config(tmp_path, algorithm)
    grid = create_grid(config)
    algo = SIMPLE(grid)
    regular = algo.solvers["p"]
    final = create_solver(config.simulator.fvSolution.solvers["p"])
    algo.solvers["pFinal"] = final
    calls = Mock()
    normal_spy = Mock(wraps=regular.solve)
    final_spy = Mock(wraps=final.solve)
    calls.attach_mock(normal_spy, "regular")
    calls.attach_mock(final_spy, "final")
    monkeypatch.setattr(regular, "solve", normal_spy)
    monkeypatch.setattr(final, "solve", final_spy)

    algo.step()

    assert [call[0] for call in calls.mock_calls] == ["regular"] * 3
    final_spy.assert_not_called()


def test_simplec_coefficient_uses_h1(tmp_path: pathlib.Path) -> None:
    """rAtU = 1 / (1/rAU - H1) with H1 = -sum(off-diagonals) / V."""
    grid = create_grid(channel_flow_config(tmp_path, _simple()))
    algo = SIMPLE(grid)
    nu_eff = algo.turbulence.nu_eff()
    UEqn = fvm.div(algo.phi, algo.U) - fvm.laplacian(nu_eff, algo.U)
    # Implicit under-relaxation only scales the diagonal.
    UEqn.diag = UEqn.diag / 0.7
    algo.rAU.data = 1.0 / UEqn.A()

    h1 = torch.zeros_like(UEqn.diag)
    h1.index_add_(0, grid.owner, -UEqn.upper)
    h1.index_add_(0, grid.neighbour, -UEqn.lower)
    h1 = h1 / grid.cell_volumes
    torch.testing.assert_close(UEqn.H1(), h1)

    rAtU = simplec_rAtU(UEqn, algo.rAU, bounded=False)
    torch.testing.assert_close(rAtU, 1.0 / (1.0 / algo.rAU.data - h1))
    # Off-diagonals of a convection-diffusion matrix are negative, so the
    # SIMPLEC coefficient is larger than rAU.
    assert bool(torch.all(rAtU > algo.rAU.data))
    bounded = simplec_rAtU(UEqn, algo.rAU, bounded=True)
    assert bool(torch.all(bounded <= 10.0 * algo.rAU.data + 1e-12))
    # Default is the bounded form used by the algorithms.
    default = simplec_rAtU(UEqn, algo.rAU)
    torch.testing.assert_close(default, bounded)


def test_ddt_corr_vanishes_for_consistent_old_flux(
    tmp_path: pathlib.Path,
) -> None:
    """
    Consistent history needs no correction; a flux mismatch gets a bounded
    one.
    """
    grid = create_grid(channel_flow_config(tmp_path, _simple()))
    U = CellField(grid, "U_ddt", FieldRole.TRANSIENT, (3,))
    phi = FaceField(grid, "phi_ddt", FieldRole.LOCAL, ())
    single = phi.single_mask

    # Hanging faces use a skew-corrected interpolation, so restrict the
    # exactness check to a uniform field where every scheme agrees.
    U.data = torch.ones_like(U.data)
    U.update_history()
    phi.single_data = torch.sum(
        fvc.interpolate(U).single_data * grid.Sf[single], dim=1, keepdim=False
    )
    phi.update_history()
    corr = fvc.ddt_corr(U, phi)
    assert float(corr.abs().max()) < 1e-12

    # A mismatch produces a bounded correction: |ddtCorr| <= |phiCorr| / dt.
    phi.single_data = phi.single_data * 1.5
    phi.update_history()
    corr = fvc.ddt_corr(U, phi)
    phi_corr = phi.old_single_data - torch.sum(
        fvc.interpolate(U).single_data * grid.Sf[single], dim=1, keepdim=False
    )
    assert bool(torch.all(corr.abs() <= phi_corr.abs() / grid.dt + 1e-12))
    assert float(corr.abs().max()) > 0.0

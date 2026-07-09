import logging

import torch

from gridfoam.algorithms.base import AlgorithmBase
from gridfoam.algorithms.utils.pressure_correction import (
    correct_phi,
    correct_velocity,
    finalize_pressure_correction,
    solve_pressure_poisson,
)
from gridfoam.algorithms.utils.reference_value import needs_reference_value
from gridfoam.algorithms.utils.residual import (
    continuity_residual,
    field_initial_residual,
    residual_satisfied,
)
from gridfoam.core.equation import equation
from gridfoam.core.field import (
    get_or_create_cellfield,
    get_or_create_facefield,
)
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.name import make_field_name
from gridfoam.fv import fvc, fvm
from gridfoam.fv.flux import compute_phi_hbya, correct_flux
from gridfoam.meta.config import SIMPLEAlgorithm, normalize_residual_control
from gridfoam.meta.enums import FieldRole
from gridfoam.models.turbulence.base import TurbulenceModel
from gridfoam.models.turbulence.factory import create_turbulence_model
from gridfoam.solvers.base import SolveStats
from gridfoam.solvers.factory import create_solver
from gridfoam.solvers.resolver import resolve_solver

logger = logging.getLogger(__name__)


class SIMPLE(AlgorithmBase):
    """
    SIMPLE (Semi-Implicit Method for Pressure Linked Equations).

    Solves steady incompressible Navier-Stokes equations with
    under-relaxation and double-sided immersed-boundary support.

    Each ``step()`` is one pseudo-time iteration. When ``residualControl`` is
    configured, ``has_converged()`` reports whether all monitored fields
    satisfy their tolerances and the runner may stop early.

    Parameters
    ----------
    grid : IGridBase
        Computational grid.
    phase : str | None, optional
        Phase name. Default is None.
    """

    def __init__(
        self,
        grid: IGridBase,
        phase: str | None = None,
    ):
        self._grid = grid

        U_name = make_field_name("U", phase=phase)
        p_name = make_field_name("p", phase=phase)
        phi_name = make_field_name("phi", phase=phase)
        rAU_name = make_field_name("rAU", phase=phase)
        HbyA_name = make_field_name("HbyA", phase=phase)

        self.U = get_or_create_cellfield(grid, U_name, FieldRole.LOCAL, 3)
        self.p = get_or_create_cellfield(grid, p_name, FieldRole.LOCAL, 1)
        self.phi = get_or_create_facefield(grid, phi_name, FieldRole.LOCAL, 1)
        self.rAU = get_or_create_cellfield(grid, rAU_name, FieldRole.LOCAL, 1)
        self.HbyA = get_or_create_cellfield(grid, HbyA_name, FieldRole.LOCAL, 3)

        self.solvers = {
            field_name: create_solver(config)
            for field_name, config in (
                self.grid.sim_config.fvSolution.solvers.items()
            )
        }
        self._turbulence = create_turbulence_model(grid)

        algorithm_config = grid.sim_config.fvSolution.algorithm
        assert isinstance(algorithm_config, SIMPLEAlgorithm)
        self._algorithm_config = algorithm_config
        self.alpha_U = algorithm_config.relaxationFactors.equations[U_name]
        self.alpha_p = algorithm_config.relaxationFactors.equations[p_name]
        self.n_non_orthogonal_correctors = (
            algorithm_config.nNonOrthogonalCorrectors
        )
        self.consistent = algorithm_config.consistent
        self.adjust_phi_enabled = grid.sim_config.fvSolution.adjustPhi
        self._residual_control = normalize_residual_control(
            algorithm_config.residualControl
        )
        self._initial_residuals: dict[str, float] = {}
        self._current_residuals: dict[str, float] = {}

        self.p_needs_ref = needs_reference_value(self.p)
        if self.p_needs_ref:
            if algorithm_config.pRefCell is None:
                raise ValueError(
                    "pRefCell is required when p_needs_ref is True"
                )
            self.p_ref_cell = algorithm_config.pRefCell
            if algorithm_config.pRefValue is None:
                raise ValueError(
                    "pRefValue is required when p_needs_ref is True"
                )
            self.p_ref_value = algorithm_config.pRefValue

        correct_flux(self.phi, self.U, update_internal=True)

    @property
    def grid(self) -> IGridBase:
        return self._grid

    @property
    def turbulence(self) -> TurbulenceModel:
        return self._turbulence

    def has_converged(self) -> bool:
        """
        Return whether all configured ``residualControl`` fields are satisfied.

        Returns
        -------
        bool
            ``True`` when every monitored field meets absolute and relative
            tolerances. Returns ``False`` if ``residualControl`` is empty.
        """
        if not self._residual_control:
            return False

        for field_name, entry in self._residual_control.items():
            residual = self._current_residuals.get(field_name)
            initial = self._initial_residuals.get(field_name)
            if residual is None or initial is None:
                return False
            if not residual_satisfied(
                residual,
                entry.tolerance,
                entry.rel_tolerance,
                initial,
            ):
                return False
        return True

    def _record_residual(
        self,
        field_name: str,
        residual: float,
    ) -> None:
        """Store the current residual and capture the first value as initial."""
        if field_name not in self._initial_residuals:
            self._initial_residuals[field_name] = residual
        self._current_residuals[field_name] = residual

    def step(self):
        grid = self.grid
        logger.info("SIMPLE step start")
        solve_stats: dict[str, tuple[SolveStats, ...]] = {}

        # =========================================================
        # Momentum predictor
        # =========================================================
        # Effective viscosity (nu + nu_t)
        nu_eff = self.turbulence.nu_eff()

        # Steady momentum equation (without ddt term):
        # div(phi, U) - laplacian(nu_eff, U)
        UEqn_mat = fvm.div(self.phi, self.U) - fvm.laplacian(nu_eff, self.U)

        # Velocity under-relaxation
        # OpenFOAM-style relaxation:
        # A_new = A / alpha,
        # source_new = source + (1-alpha)/alpha * A * U_old
        A_old = UEqn_mat.diag.clone()
        UEqn_mat.diag = A_old / self.alpha_U

        relax_source = (
            ((1.0 - self.alpha_U) / self.alpha_U) * A_old * self.U.data
        )
        UEqn_mat.source = UEqn_mat.source + relax_source

        # Keep the original source without pressure-gradient contribution
        original_source = UEqn_mat.source.clone()

        # Add pressure-gradient source term to RHS (-grad(p) * V)
        grad_p = fvc.grad(self.p)
        UEqn_mat.source = UEqn_mat.source - grad_p.data * grid.cell_volumes

        if self.U.name in self._residual_control:
            self._record_residual(
                self.U.name,
                field_initial_residual(UEqn_mat, self.U),
            )

        # Solve the momentum predictor equation (obtain U*)
        momentum_eq = equation(self.U, UEqn_mat)
        u_result = self.solvers[momentum_eq.name].solve(momentum_eq)
        self.U.data = u_result.solution
        solve_stats[self.U.name] = u_result.stats

        # =========================================================
        # Pressure equation assembly (HbyA and rAU)
        # =========================================================
        self.rAU.data = 1.0 / UEqn_mat.A()

        # Compute HbyA with H() using source without pressure gradient
        UEqn_mat.source = original_source
        self.HbyA.data = UEqn_mat.H(self.U.data) * self.rAU.data

        # Interpolate HbyA to faces and compute initial flux phi_HbyA
        phi_hbya = compute_phi_hbya(self.phi, self.HbyA)

        # =========================================================
        # Pressure Poisson equation
        # =========================================================
        # -∇・(rAU ∇p) = -∇・U*

        logger.debug(
            "SIMPLE continuity residual L2=%.3e",
            continuity_residual(self.phi),
        )

        # Store old pressure for pressure under-relaxation
        p_old = self.p.data.clone()

        # solve pressure Poisson equation (uses pFinal when configured)
        div_phi = fvc.div(self.phi).data
        p_solver = resolve_solver(self.solvers, self.p.name, is_final=True)
        p_eqn_mat, p_stats = solve_pressure_poisson(
            self.p,
            self.rAU,
            div_phi,
            p_solver,
            n_non_orthogonal_correctors=self.n_non_orthogonal_correctors,
            p_needs_ref=self.p_needs_ref,
            p_ref_cell=self.p_ref_cell if self.p_needs_ref else None,
            p_ref_value=self.p_ref_value if self.p_needs_ref else None,
        )
        solve_stats[self.p.name] = p_stats

        if self.p.name in self._residual_control:
            self._record_residual(
                self.p.name,
                field_initial_residual(p_eqn_mat, self.p),
            )

        # flux correction with the unrelaxed pressure-equation solution.
        correct_phi(
            self.phi,
            self.rAU,
            self.p,
            consistent=self.consistent,
            phi_hbya=phi_hbya,
            p_eqn_mat=p_eqn_mat,
        )

        # apply pressure relaxation
        self.p.data = p_old + self.alpha_p * (self.p.data - p_old)

        # =========================================================
        # Velocity and flux correction
        # =========================================================
        correct_velocity(self.U, self.HbyA, self.rAU, self.p)
        finalize_pressure_correction(
            self.phi,
            self.U,
            adjust_phi_enabled=self.adjust_phi_enabled,
        )  # boundary phi sync + optional adjustPhi

        if "phi" in self._residual_control:
            self._record_residual("phi", continuity_residual(self.phi))

        logger.debug(
            "SIMPLE corrected flux L2=%.3e",
            torch.linalg.vector_norm(self.phi.single_data, ord=2).item(),
        )

        # =========================================================
        # Turbulence model update
        # =========================================================
        self.turbulence.correct(self.U, self.phi)
        self._finalize_diagnostics(self.phi, solve_stats)
        logger.info("SIMPLE step end")

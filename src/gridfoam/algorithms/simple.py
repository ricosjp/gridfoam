import logging

import torch

from gridfoam.algorithms.base import AlgorithmBase
from gridfoam.algorithms.utils.pressure_correction import (
    apply_simplec,
    correct_phi,
    correct_velocity,
    simplec_rAtU,
    solve_pressure_poisson,
)
from gridfoam.algorithms.utils.reference_value import needs_reference_value
from gridfoam.algorithms.utils.residual import (
    continuity_residual,
    field_initial_residual,
    residual_satisfied,
)
from gridfoam.core.dimensions import (
    DIM_KIN_PRESSURE,
    DIM_RAU,
    DIM_VELOCITY,
    DIM_VOL_FLUX,
)
from gridfoam.core.equation import equation
from gridfoam.core.field import (
    CellField,
    FaceField,
    get_or_create_cellfield,
    get_or_create_facefield,
)
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.name import make_field_name
from gridfoam.fv import fvc, fvm
from gridfoam.fv.adjust_phi import adjust_phi
from gridfoam.fv.flux import compute_phi_hbya, correct_flux
from gridfoam.meta.config import SIMPLEAlgorithm, normalize_residual_control
from gridfoam.meta.enums import FieldRole
from gridfoam.models.turbulence.base import TurbulenceModel
from gridfoam.models.turbulence.factory import create_turbulence_model
from gridfoam.solvers.base import LinearSolver, SolveStats
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

    Attributes
    ----------
    U : CellField
        Velocity field with shape ``[C, 3]``.
    p : CellField
        Pressure field with shape ``[C]``.
    phi : FaceField
        Volumetric face flux.
    phi_hbya : FaceField
        Predicted flux ``flux(constrainHbyA(HbyA))`` (all face blocks).
    rAU : CellField
        Reciprocal of the momentum diagonal, ``1/A(U)``.
    rAtU : CellField
        Pressure-equation coefficient: ``rAU`` or, with ``consistent``,
        the SIMPLEC value ``1/(1/A(U) - H1)``.
    HbyA : CellField
        Explicit momentum contribution ``H(U)/A(U)``.
    solvers : dict[str, LinearSolver]
        Linear solvers keyed by field name from ``fvSolution``.
    alpha_U : float
        Under-relaxation factor for the momentum equation.
    alpha_p : float
        Under-relaxation factor for the pressure equation.
    n_non_orthogonal_correctors : int
        Number of non-orthogonal pressure correctors.
    consistent : bool
        Whether the SIMPLEC formulation (OpenFOAM ``consistent yes``) is
        enabled.
    adjust_phi_enabled : bool
        Whether ``adjustPhi`` continuity adjustment is enabled.
    p_needs_ref : bool
        Whether pressure requires a reference cell/value.
    p_ref_cell : int
        Reference pressure cell index when ``p_needs_ref`` is True.
    p_ref_value : float
        Reference pressure value when ``p_needs_ref`` is True.
    grid : IGridBase
        Computational grid owned by the algorithm.
    turbulence : TurbulenceModel
        Turbulence model used to evaluate effective viscosity.
    """

    U: CellField
    """Velocity field with shape ``[C, 3]``."""

    p: CellField
    """Pressure field with shape ``[C]``."""

    phi: FaceField
    """Volumetric face flux."""

    phi_hbya: FaceField
    """Predicted flux ``flux(constrainHbyA(HbyA))``."""

    rAU: CellField
    """Reciprocal of the momentum diagonal, ``1/A(U)``."""

    rAtU: CellField
    """Pressure-equation coefficient (``rAU`` or the SIMPLEC value)."""

    HbyA: CellField
    """Explicit momentum contribution ``H(U)/A(U)``."""

    alpha_U: float
    """Under-relaxation factor for the momentum equation."""

    alpha_p: float
    """Under-relaxation factor for the pressure equation."""

    n_non_orthogonal_correctors: int
    """Number of non-orthogonal pressure correctors."""

    consistent: bool
    """Whether the SIMPLEC formulation is enabled."""

    adjust_phi_enabled: bool
    """Whether ``adjustPhi`` continuity adjustment is enabled."""

    p_needs_ref: bool
    """Whether pressure requires a reference cell/value."""

    def __init__(
        self,
        grid: IGridBase,
        phase: str | None = None,
    ):
        self._grid = grid

        U_name = make_field_name("U", phase=phase)
        p_name = make_field_name("p", phase=phase)
        phi_name = make_field_name("phi", phase=phase)
        phi_hbya_name = make_field_name("phiHbyA", phase=phase)
        rAU_name = make_field_name("rAU", phase=phase)
        rAtU_name = make_field_name("rAtU", phase=phase)
        HbyA_name = make_field_name("HbyA", phase=phase)

        self.U = get_or_create_cellfield(
            grid, U_name, FieldRole.LOCAL, (3,), dimension=DIM_VELOCITY
        )
        self.p = get_or_create_cellfield(
            grid, p_name, FieldRole.LOCAL, (), dimension=DIM_KIN_PRESSURE
        )
        self.phi = get_or_create_facefield(
            grid, phi_name, FieldRole.LOCAL, (), dimension=DIM_VOL_FLUX
        )
        self.phi_hbya = get_or_create_facefield(
            grid,
            phi_hbya_name,
            FieldRole.LOCAL,
            (),
            dimension=DIM_VOL_FLUX,
            export=False,
        )
        self.rAU = get_or_create_cellfield(
            grid, rAU_name, FieldRole.LOCAL, (), dimension=DIM_RAU
        )
        self.rAtU = get_or_create_cellfield(
            grid, rAtU_name, FieldRole.LOCAL, (), dimension=DIM_RAU
        )
        self.HbyA = get_or_create_cellfield(
            grid, HbyA_name, FieldRole.LOCAL, (3,), dimension=DIM_VELOCITY
        )

        self._solvers = {
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
        """Computational grid owned by the algorithm."""
        return self._grid

    @property
    def turbulence(self) -> TurbulenceModel:
        """Turbulence model used to evaluate effective viscosity."""
        return self._turbulence

    @property
    def solvers(self) -> dict[str, LinearSolver]:
        """Linear solvers keyed by field name from ``fvSolution``."""
        return self._solvers

    def has_converged(self) -> bool:
        """
        Return whether all configured ``residualControl`` fields are satisfied.

        Returns
        -------
        bool
            ``True`` when every monitored field is below ``tolerance``.
            ``rel_tolerance`` is ignored. Returns ``False`` if
            ``residualControl`` is empty.
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
                0.0,
                initial,
            ):
                return False
        return True

    def has_simulation_converged(self) -> bool:
        """A converged steady SIMPLE iteration ends the simulation."""
        return self.has_converged()

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

        relax_coeff = ((1.0 - self.alpha_U) / self.alpha_U) * A_old
        relax_source = relax_coeff[:, None] * self.U.data
        UEqn_mat.source = UEqn_mat.source + relax_source

        # Keep the original source without pressure-gradient contribution
        original_source = UEqn_mat.source.clone()

        # Add pressure-gradient source term to RHS (-grad(p) * V)
        grad_p = fvc.grad(self.p)
        UEqn_mat.source = (
            UEqn_mat.source - grid.cell_volumes[:, None] * grad_p.data
        )

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
        # Pressure equation assembly (HbyA, rAU, phiHbyA)
        # =========================================================
        self.rAU.data = 1.0 / UEqn_mat.A()

        # Compute HbyA with H() using source without pressure gradient
        UEqn_mat.source = original_source
        self.HbyA.data = self.rAU.data[:, None] * UEqn_mat.H(self.U.data)

        # phiHbyA = flux(constrainHbyA(HbyA, U)) on every face block
        compute_phi_hbya(self.phi_hbya, self.HbyA, self.U)
        if self.adjust_phi_enabled:
            adjust_phi(self.phi_hbya, self.U, self.p)

        # SIMPLEC: rAtU = 1/(1/rAU - H1) and consistent phiHbyA / HbyA
        if self.consistent:
            self.rAtU.data = simplec_rAtU(UEqn_mat, self.rAU)
            apply_simplec(self.phi_hbya, self.HbyA, self.p, self.rAU, self.rAtU)
        else:
            self.rAtU.data = self.rAU.data

        # =========================================================
        # Pressure Poisson equation
        # =========================================================
        # -∇・(rAtU ∇p) = -∇・phiHbyA

        logger.debug(
            "SIMPLE predicted-flux divergence L2=%.3e",
            continuity_residual(self.phi_hbya),
        )

        # Store old pressure for pressure under-relaxation
        p_old = self.p.data.clone()

        # solve pressure Poisson equation (uses pFinal when configured)
        div_phi_hbya = fvc.div(self.phi_hbya).data
        p_solver = resolve_solver(self.solvers, self.p.name, is_final=True)
        p_result = solve_pressure_poisson(
            self.p,
            self.rAtU,
            div_phi_hbya,
            p_solver,
            n_non_orthogonal_correctors=self.n_non_orthogonal_correctors,
            p_needs_ref=self.p_needs_ref,
            p_ref_cell=self.p_ref_cell if self.p_needs_ref else None,
            p_ref_value=self.p_ref_value if self.p_needs_ref else None,
        )
        solve_stats[self.p.name] = p_result.stats

        if self.p.name in self._residual_control:
            self._record_residual(self.p.name, p_result.initial_residual)

        # phi = phiHbyA - pEqn.flux() with the unrelaxed pressure solution.
        correct_phi(self.phi, self.phi_hbya, p_result.matrix, self.p, self.rAtU)

        # apply pressure relaxation
        self.p.data = p_old + self.alpha_p * (self.p.data - p_old)

        # =========================================================
        # Velocity correction
        # =========================================================
        correct_velocity(self.U, self.HbyA, self.rAtU, self.p)

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

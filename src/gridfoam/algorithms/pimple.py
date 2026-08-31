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
from gridfoam.fv.flux import compute_phi_hbya, correct_flux
from gridfoam.meta.config import PIMPLEAlgorithm, normalize_residual_control
from gridfoam.meta.enums import FieldRole
from gridfoam.models.turbulence.base import TurbulenceModel
from gridfoam.models.turbulence.factory import create_turbulence_model
from gridfoam.solvers.base import LinearSolver, SolveStats
from gridfoam.solvers.factory import create_solver
from gridfoam.solvers.resolver import resolve_solver

logger = logging.getLogger(__name__)


class PIMPLE(AlgorithmBase):
    """
    PIMPLE (PISO + SIMPLE).

    Provides stable transient calculations with large time steps by
    introducing an outer corrector loop around the PISO loop.

    The outer loop may exit early when ``residualControl`` is satisfied.
    The final inner corrector uses the ``pFinal`` solver when configured.

    Parameters
    ----------
    grid : IGridBase
        Computational grid.
    phase : str | None, optional
        Phase name. Default is None.

    Attributes
    ----------
    U : CellField
        Velocity field with shape ``[C, 3]`` (``FieldRole.TRANSIENT``).
    p : CellField
        Pressure field with shape ``[C, 1]``.
    phi : FaceField
        Volumetric face flux.
    rAU : CellField
        Reciprocal of the momentum diagonal, ``1/A(U)``.
    HbyA : CellField
        Explicit momentum contribution ``H(U)/A(U)``.
    solvers : dict[str, LinearSolver]
        Linear solvers keyed by field name from ``fvSolution``.
    n_outer_correctors : int
        Number of PIMPLE outer correctors.
    n_correctors : int
        Number of inner PISO pressure--velocity correctors.
    n_non_orthogonal_correctors : int
        Number of non-orthogonal pressure correctors.
    consistent : bool
        Whether SIMPLEC-style consistent flux correction is enabled.
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
    """Velocity field with shape ``[C, 3]`` (``FieldRole.TRANSIENT``)."""

    p: CellField
    """Pressure field with shape ``[C, 1]``."""

    phi: FaceField
    """Volumetric face flux."""

    rAU: CellField
    """Reciprocal of the momentum diagonal, ``1/A(U)``."""

    HbyA: CellField
    """Explicit momentum contribution ``H(U)/A(U)``."""

    solvers: dict[str, LinearSolver]
    """Linear solvers keyed by field name from ``fvSolution``."""

    n_outer_correctors: int
    """Number of PIMPLE outer correctors."""

    n_correctors: int
    """Number of inner PISO pressure--velocity correctors."""

    n_non_orthogonal_correctors: int
    """Number of non-orthogonal pressure correctors."""

    consistent: bool
    """Whether SIMPLEC-style consistent flux correction is enabled."""

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
        rAU_name = make_field_name("rAU", phase=phase)
        HbyA_name = make_field_name("HbyA", phase=phase)

        self.U = get_or_create_cellfield(
            grid, U_name, FieldRole.TRANSIENT, 3, dimension=DIM_VELOCITY
        )
        self.p = get_or_create_cellfield(
            grid, p_name, FieldRole.LOCAL, 1, dimension=DIM_KIN_PRESSURE
        )
        self.phi = get_or_create_facefield(
            grid, phi_name, FieldRole.LOCAL, 1, dimension=DIM_VOL_FLUX
        )
        self.rAU = get_or_create_cellfield(
            grid, rAU_name, FieldRole.LOCAL, 1, dimension=DIM_RAU
        )
        self.HbyA = get_or_create_cellfield(
            grid, HbyA_name, FieldRole.LOCAL, 3, dimension=DIM_VELOCITY
        )

        self.solvers = {
            field_name: create_solver(config)
            for field_name, config in (
                self.grid.sim_config.fvSolution.solvers.items()
            )
        }
        self._turbulence = create_turbulence_model(grid)

        algorithm_config = grid.sim_config.fvSolution.algorithm
        assert isinstance(algorithm_config, PIMPLEAlgorithm)
        self.n_outer_correctors = algorithm_config.nOuterCorrectors
        self.n_correctors = algorithm_config.nCorrectors
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
        self._outer_converged = False

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

    def has_converged(self) -> bool:
        """
        Return whether the latest outer corrector satisfied residual control.

        Returns
        -------
        bool
            ``True`` when every monitored field meets absolute and relative
            tolerances after an outer corrector.
        """
        return self._outer_converged

    def _record_residual(
        self,
        field_name: str,
        residual: float,
    ) -> None:
        """Store the current residual and capture the first value as initial."""
        if field_name not in self._initial_residuals:
            self._initial_residuals[field_name] = residual
        self._current_residuals[field_name] = residual

    def _check_outer_convergence(self) -> bool:
        """
        Return whether all configured ``residualControl`` fields are satisfied.

        Returns
        -------
        bool
            ``True`` when every monitored field meets its tolerances.
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

    def step(self):
        grid = self.grid
        logger.info(
            "PIMPLE step start outer=%d inner=%d",
            self.n_outer_correctors,
            self.n_correctors,
        )

        solve_stats: dict[str, tuple[SolveStats, ...]] = {}
        self._outer_converged = False
        # =========================================================
        # PIMPLE outer loop
        # =========================================================
        for outer in range(self.n_outer_correctors):
            logger.debug(
                "PIMPLE outer loop %d/%d",
                outer + 1,
                self.n_outer_correctors,
            )
            # Momentum predictor
            nu_eff = self.turbulence.nu_eff()
            UEqn_mat = (
                fvm.ddt(self.U)
                + fvm.div(self.phi, self.U)
                - fvm.laplacian(nu_eff, self.U)
            )

            # Keep the original source without pressure-gradient contribution
            original_source = UEqn_mat.source.clone()

            # Add pressure-gradient source term (-grad(p) * V)
            grad_p = fvc.grad(self.p)
            UEqn_mat.source = original_source - grad_p.data * grid.cell_volumes

            # Solve momentum predictor (obtain U*)
            if self.U.name in self._residual_control and outer == 0:
                self._record_residual(
                    self.U.name,
                    field_initial_residual(UEqn_mat, self.U),
                )

            momentum_eq = equation(self.U, UEqn_mat)
            u_result = self.solvers[momentum_eq.name].solve(momentum_eq)
            self.U.data = u_result.solution
            solve_stats[self.U.name] = u_result.stats

            # =========================================================
            # PISO corrector loop (pressure-velocity coupling)
            # =========================================================
            for inner in range(self.n_correctors):
                logger.debug(
                    "PIMPLE inner loop %d/%d",
                    inner + 1,
                    self.n_correctors,
                )
                self.rAU.data = 1.0 / UEqn_mat.A()

                # Compute HbyA with H() without pressure-gradient source
                UEqn_mat.source = original_source
                self.HbyA.data = UEqn_mat.H(self.U.data) * self.rAU.data

                # Interpolate HbyA to faces and compute phi_HbyA
                phi_hbya = compute_phi_hbya(self.phi, self.HbyA)

                logger.debug(
                    "PIMPLE continuity residual L2=%.3e",
                    continuity_residual(self.phi),
                )

                # solve pressure Poisson equation (pFinal on last corrector)
                div_phi = fvc.div(self.phi).data
                is_final = inner == self.n_correctors - 1
                p_solver = resolve_solver(
                    self.solvers, self.p.name, is_final=is_final
                )
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

                if self.p.name in self._residual_control and is_final:
                    self._record_residual(
                        self.p.name,
                        field_initial_residual(p_eqn_mat, self.p),
                    )

                # flux correction
                correct_phi(
                    self.phi,
                    self.rAU,
                    self.p,
                    consistent=self.consistent,
                    phi_hbya=phi_hbya,
                    p_eqn_mat=p_eqn_mat,
                )

                # Velocity and flux correction
                correct_velocity(self.U, self.HbyA, self.rAU, self.p)
                finalize_pressure_correction(
                    self.phi,
                    self.U,
                    adjust_phi_enabled=self.adjust_phi_enabled,
                )
                logger.debug(
                    "PIMPLE corrected flux L2=%.3e",
                    torch.linalg.vector_norm(
                        self.phi.single_data, ord=2
                    ).item(),
                )

            # =========================================================
            # Turbulence model update
            # =========================================================
            self.turbulence.correct(self.U, self.phi)

            if "phi" in self._residual_control:
                self._record_residual("phi", continuity_residual(self.phi))

            if self._check_outer_convergence():
                self._outer_converged = True
                logger.info(
                    "PIMPLE outer loop converged at outer=%d",
                    outer + 1,
                )
                break

        self.U.update_history()
        self._finalize_diagnostics(self.phi, solve_stats)
        logger.info("PIMPLE step end")

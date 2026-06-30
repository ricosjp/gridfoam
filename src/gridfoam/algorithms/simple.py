import logging

import torch

from gridfoam.algorithms.base import AlgorithmBase
from gridfoam.algorithms.utils import (
    needs_reference_value,
    set_reference_value,
)
from gridfoam.core.equation import equation
from gridfoam.core.field import (
    get_or_create_cellfield,
    get_or_create_facefield,
)
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.name import make_field_name
from gridfoam.fv import fvc, fvm
from gridfoam.fv.flux import correct_flux
from gridfoam.meta.config import SIMPLEAlgorithm
from gridfoam.meta.enums import FieldRole
from gridfoam.models.turbulence.base import TurbulenceModel
from gridfoam.models.turbulence.factory import create_turbulence_model
from gridfoam.solvers.factory import create_solver

logger = logging.getLogger(__name__)


class SIMPLE(AlgorithmBase):
    """
    SIMPLE (Semi-Implicit Method for Pressure Linked Equations).

    Solves steady incompressible Navier-Stokes equations with
    under-relaxation and double-sided immersed-boundary support.

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
        self.alpha_U = algorithm_config.relaxationFactors.equations[U_name]
        self.alpha_p = algorithm_config.relaxationFactors.equations[p_name]
        self.n_non_orthogonal_correctors = (
            algorithm_config.nNonOrthogonalCorrectors
        )

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

    def step(self):
        grid = self.grid
        logger.info("SIMPLE step start")

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

        # Solve the momentum predictor equation (obtain U*)
        momentum_eq = equation(self.U, UEqn_mat)
        self.U.data = self.solvers[momentum_eq.name].solve(momentum_eq)

        # =========================================================
        # Pressure equation assembly (HbyA and rAU)
        # =========================================================
        self.rAU.data = 1.0 / UEqn_mat.A()

        # Compute HbyA with H() using source without pressure gradient
        UEqn_mat.source = original_source
        self.HbyA.data = UEqn_mat.H(self.U.data) * self.rAU.data

        # Interpolate HbyA to faces and compute initial flux phi_HbyA
        HbyA_f = fvc.interpolate(self.HbyA)

        # only update the internal faces
        # (other faces are constrained by boundary conditions)
        self.phi.single_data = torch.sum(
            HbyA_f.single_data * grid.Sf[HbyA_f.single_mask],
            dim=1,
            keepdim=True,
        )

        # =========================================================
        # Pressure Poisson equation
        # =========================================================
        # -∇・(rAU ∇p) = -∇・U*

        logger.debug(
            "SIMPLE continuity residual L2=%.3e",
            torch.linalg.vector_norm(fvc.div(self.phi).data, ord=2).item(),
        )

        # Store old pressure for pressure under-relaxation
        p_old = self.p.data.clone()

        # solve pressure Poisson equation
        div_phi = fvc.div(self.phi).data
        for corr in range(self.n_non_orthogonal_correctors + 1):
            pEqn_mat = -fvm.laplacian(self.rAU.data, self.p)
            pEqn_mat.source = pEqn_mat.source - div_phi
            if self.p_needs_ref:
                set_reference_value(pEqn_mat, self.p_ref_cell, self.p_ref_value)

            pressure_eq = equation(self.p, pEqn_mat)
            self.p.data = self.solvers[self.p.name].solve(pressure_eq)

            if corr < self.n_non_orthogonal_correctors:
                logger.debug(
                    "non-orthogonal pressure corrector %d/%d",
                    corr + 1,
                    self.n_non_orthogonal_correctors,
                )

        # flux correction with the unrelaxed pressure-equation solution.
        sn_grad_p = fvc.sn_grad(self.p)
        rAU_f = fvc.interpolate(self.rAU)

        # phi = phi_HbyA - rAU_f * |Sf| * snGrad(p)
        mag_Sf = torch.linalg.vector_norm(
            grid.Sf[self.phi.single_mask], dim=1, keepdim=True
        )
        self.phi.single_data = (
            self.phi.single_data
            - rAU_f.single_data * mag_Sf * sn_grad_p.single_data
        )

        # apply pressure relaxation
        self.p.data = p_old + self.alpha_p * (self.p.data - p_old)

        # =========================================================
        # Velocity and flux correction
        # =========================================================
        grad_p = fvc.grad(self.p)

        # Velocity correction: U = HbyA - rAU * grad(p)
        self.U.data = self.HbyA.data - self.rAU.data * grad_p.data

        correct_flux(self.phi, self.U)
        logger.debug(
            "SIMPLE corrected flux L2=%.3e",
            torch.linalg.vector_norm(self.phi.single_data, ord=2).item(),
        )

        # =========================================================
        # Turbulence model update
        # =========================================================
        self.turbulence.correct(self.U, self.phi)
        logger.info("SIMPLE step end")

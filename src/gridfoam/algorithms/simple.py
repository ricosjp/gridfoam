import logging

import torch

from gridfoam.algorithms.base import AlgorithmBase
from gridfoam.algorithms.utils import (
    needs_reference_value,
    solve_pressure_poisson,
)
from gridfoam.core.builtins import make_builtin_key
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.base import IGridBase
from gridfoam.fv import fvc, fvm
from gridfoam.fv.flux import correct_flux
from gridfoam.meta.enums import FieldRole
from gridfoam.models.turbulence.base import TurbulenceModel
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
    U : CellField
        Velocity field.
    p : CellField
        Pressure field.
    turbulence : TurbulenceModel
        Turbulence model.
    alpha_U : float, optional
        Under-relaxation factor for velocity. Default is 0.7.
    alpha_p : float, optional
        Under-relaxation factor for pressure. Default is 0.3.
    """

    def __init__(
        self,
        grid: IGridBase,
        U: CellField,
        p: CellField,
        turbulence: TurbulenceModel,
        alpha_U: float = 0.7,
        alpha_p: float = 0.3,
    ):
        self.grid = grid
        self.U = U
        self.p = p
        builtin_scope = "global"
        phi_builtin_key = make_builtin_key("phi", scope=builtin_scope)
        builtin_phi = self.grid.get_builtin_field(phi_builtin_key)
        if isinstance(builtin_phi, FaceField):
            self.phi = builtin_phi
        else:
            self.phi = FaceField(
                grid=self.grid,
                name="phi",
                role=FieldRole.LOCAL,
                num_components=1,
                export=False,
            )
        self.solvers = {
            eq_name: create_solver(config)
            for eq_name, config in (
                self.grid.sim_config.fvSolution.solvers.items()
            )
        }
        self.turbulence = turbulence
        self.alpha_U = alpha_U
        self.alpha_p = alpha_p
        self.grid.register_builtin_field(
            make_builtin_key("U", scope=builtin_scope), self.U
        )
        self.grid.register_builtin_field(
            make_builtin_key("p", scope=builtin_scope), self.p
        )
        self.grid.register_builtin_field(
            make_builtin_key("phi", scope=builtin_scope), self.phi
        )

        self.rAU_field = CellField(
            grid,
            name="rAU",
            role=FieldRole.LOCAL,
            num_components=1,
            export=False,
        )
        self.grid.register_builtin_field(
            make_builtin_key("rAU", scope=builtin_scope), self.rAU_field
        )

        self.HbyA_field = CellField(
            grid,
            name="HbyA",
            role=FieldRole.LOCAL,
            num_components=3,
            export=False,
        )
        self.grid.register_builtin_field(
            make_builtin_key("HbyA", scope=builtin_scope), self.HbyA_field
        )

        self.p_needs_ref = needs_reference_value(self.p)
        correct_flux(self.phi, self.U, update_internal=True)
        self._single_Sf = self.grid.Sf[self.phi.single_mask]
        self._single_mag_Sf = torch.linalg.vector_norm(
            self._single_Sf, dim=1, keepdim=True
        )

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
        momentum_eq = equation("momentum", self.U, UEqn_mat)
        self.U.data = self.solvers[momentum_eq.name].solve(momentum_eq)

        # =========================================================
        # Pressure equation assembly (HbyA and rAU)
        # =========================================================
        self.rAU_field.data = 1.0 / UEqn_mat.A()

        # Compute HbyA with H() using source without pressure gradient
        UEqn_mat.source = original_source
        self.HbyA_field.data = UEqn_mat.H(self.U.data) * self.rAU_field.data

        # Interpolate HbyA to faces and compute initial flux phi_HbyA
        HbyA_f = fvc.interpolate(self.HbyA_field)

        # only update the internal faces
        # (other faces are constrained by boundary conditions)
        self.phi.single_data = torch.sum(
            HbyA_f.single_data * self._single_Sf,
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
        fv_solution = grid.sim_config.fvSolution
        p_solved = solve_pressure_poisson(
            self.p,
            self.rAU_field,
            self.phi,
            self.solvers["pressure_poisson"],
            p_needs_ref=self.p_needs_ref,
            n_non_orthogonal_correctors=fv_solution.n_non_orthogonal_correctors,
        )

        # flux correction with the unrelaxed pressure-equation solution.
        self.p.data = p_solved
        sn_grad_p = fvc.sn_grad(self.p)
        rAU_f = fvc.interpolate(self.rAU_field)

        # phi = phi_HbyA - rAU_f * |Sf| * snGrad(p)
        self.phi.single_data = (
            self.phi.single_data
            - rAU_f.single_data * self._single_mag_Sf * sn_grad_p.single_data
        )

        # apply pressure relaxation
        self.p.data = p_old + self.alpha_p * (p_solved - p_old)

        # =========================================================
        # Velocity and flux correction
        # =========================================================
        grad_p = fvc.grad(self.p)

        # Velocity correction: U = HbyA - rAU * grad(p)
        self.U.data = self.HbyA_field.data - self.rAU_field.data * grad_p.data

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

import logging

import torch

from gridfoam.algorithms.base import AlgorithmBase
from gridfoam.algorithms.utils import (
    needs_reference_value,
    set_reference_value,
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


class PIMPLE(AlgorithmBase):
    """
    PIMPLE (PISO + SIMPLE).

    Provides stable transient calculations with large time steps by
    introducing an outer corrector loop around the PISO loop.

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
    n_outer_correctors : int, optional
        Number of outer correctors. Default is 1.
    n_correctors : int, optional
        Number of inner correctors. Default is 2.
    """

    def __init__(
        self,
        grid: IGridBase,
        U: CellField,
        p: CellField,
        turbulence: TurbulenceModel,
        n_outer_correctors: int = 1,
        n_correctors: int = 2,
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
        self.n_outer_correctors = n_outer_correctors
        self.n_correctors = n_correctors
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

    def step(self):
        grid = self.grid
        logger.info(
            "PIMPLE step start outer=%d inner=%d",
            self.n_outer_correctors,
            self.n_correctors,
        )

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
            momentum_eq = equation("momentum", self.U, UEqn_mat)
            self.U.data = self.solvers[momentum_eq.name].solve(momentum_eq)

            # =========================================================
            # PISO corrector loop (pressure-velocity coupling)
            # =========================================================
            for inner in range(self.n_correctors):
                logger.debug(
                    "PIMPLE inner loop %d/%d",
                    inner + 1,
                    self.n_correctors,
                )
                self.rAU_field.data = 1.0 / UEqn_mat.A()

                # Compute HbyA with H() without pressure-gradient source
                UEqn_mat.source = original_source
                self.HbyA_field.data = (
                    UEqn_mat.H(self.U.data) * self.rAU_field.data
                )

                # Interpolate HbyA to faces and compute phi_HbyA
                HbyA_f = fvc.interpolate(self.HbyA_field)

                # only update the internal faces
                # (other faces are constrained by boundary conditions)
                self.phi.single_data = torch.sum(
                    HbyA_f.single_data * grid.Sf[HbyA_f.single_mask],
                    dim=1,
                    keepdim=True,
                )

                # Pressure Poisson equation
                pEqn_mat = -fvm.laplacian(self.rAU_field.data, self.p)
                div_phi = fvc.div(self.phi)
                logger.debug(
                    "PIMPLE continuity residual L2=%.3e",
                    torch.linalg.vector_norm(div_phi.data, ord=2).item(),
                )
                pEqn_mat.source = pEqn_mat.source - div_phi.data

                if self.p_needs_ref:
                    set_reference_value(pEqn_mat)

                # Solve pressure equation
                pressure_eq = equation("pressure_poisson", self.p, pEqn_mat)
                self.p.data = self.solvers[pressure_eq.name].solve(pressure_eq)

                # Velocity and flux correction
                grad_p = fvc.grad(self.p)
                sn_grad_p = fvc.sn_grad(self.p)
                rAU_f = fvc.interpolate(self.rAU_field)

                # Velocity correction: U = HbyA - rAU * grad(p)
                self.U.data = (
                    self.HbyA_field.data - self.rAU_field.data * grad_p.data
                )

                # Face-flux correction:
                # phi = phi_HbyA - rAU_f * |Sf| * snGrad(p)
                mag_Sf = torch.linalg.vector_norm(
                    grid.Sf[self.phi.single_mask], dim=1, keepdim=True
                )
                self.phi.single_data = (
                    self.phi.single_data
                    - rAU_f.single_data * mag_Sf * sn_grad_p.single_data
                )

                correct_flux(self.phi, self.U)
            # =========================================================
            # Turbulence model update
            # =========================================================
            self.turbulence.correct(self.U, self.phi)
        logger.info("PIMPLE step end")

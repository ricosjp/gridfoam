import logging

import torch

from gridfoam.algorithms.utils import needs_reference_value, set_reference_value
from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.core.builtins import make_builtin_key
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.base import IGridBase
from gridfoam.fv import fvc, fvm
from gridfoam.fv.flux import (
    correct_flux,
    reconstruct_U_from_phi,
)
from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import BoundaryConditionType, FieldRole
from gridfoam.meta.types import PatchName
from gridfoam.solvers.base import LinearSolver
from gridfoam.solvers.factory import create_solver

logger = logging.getLogger(__name__)


class PotentialFlow:
    def __init__(
        self,
        grid: IGridBase,
        U: CellField,
        p: CellField,
        phi: FaceField | None = None,
    ):
        potential_flow_config = grid.sim_config.fvSolution.potential_flow
        if potential_flow_config is None:
            raise ValueError(
                "potential_flow configuration in fvSolution is required."
            )

        if phi is None:
            builtin_key = make_builtin_key("phi", scope="global")
            builtin_phi = grid.get_builtin_field(builtin_key)
            if isinstance(builtin_phi, FaceField):
                phi = builtin_phi
            else:
                phi = FaceField(
                    grid=grid,
                    name="phi",
                    role=FieldRole.LOCAL,
                    num_components=1,
                    export=False,
                )
                grid.register_builtin_field(builtin_key, phi)

        boundary_conditions = grid.sim_config.boundaryConditions
        if boundary_conditions is None:
            raise ValueError(
                "boundaryConditions is required for potential flow."
            )
        p_bc_configs = boundary_conditions.get(p.name)
        if p_bc_configs is None:
            raise ValueError("pressure boundaryConditions are required.")

        self.U = U
        self.p = p
        self.phi = phi
        self.potential_flow_config = potential_flow_config
        self.solver_configs = grid.sim_config.fvSolution.solvers
        self.p_bc_configs = p_bc_configs

    def _set_Phi_bcs(self, Phi: CellField):
        bcs: dict[PatchName, BoundaryCondition] = {}
        dtype = self.U.grid.dtype
        device = self.U.grid.device

        for bc_config in self.p_bc_configs:
            if bc_config.type == BoundaryConditionType.DIRICHLET:
                for patch in bc_config.patches:
                    value = torch.tensor(
                        bc_config.value, dtype=dtype, device=device
                    )
                    bcs[patch] = DirichletBC(value)
            else:
                for patch in bc_config.patches:
                    zero_grad = torch.zeros((1,), dtype=dtype, device=device)
                    bcs[patch] = NeumannBC(zero_grad)
        Phi.add_boundary_conditions(bcs)

    def _select_solver(
        self,
        solvers: dict[str, SolverConfig],
        primary_key: str,
        fallback_key: str,
    ) -> LinearSolver:
        if primary_key in solvers:
            return create_solver(solvers[primary_key])
        return create_solver(solvers[fallback_key])

    def solve(self):
        logger.info("potential flow solve start")

        # calculate flux from velocity field
        correct_flux(self.phi, self.U, update_internal=True)
        div_phi = fvc.div(self.phi).data
        continuity_residual = torch.linalg.vector_norm(div_phi, ord=2).item()
        logger.info(
            "potential flow initial continuity residual L2=%.3e",
            continuity_residual,
        )

        # create velocity potential field
        Phi = CellField(
            self.U.grid,
            name="Phi",
            role=FieldRole.LOCAL,
            num_components=1,
            export=False,
            ref_cell_id=self.p.ref_cell_id,
            ref_value=0.0,
        )
        self._set_Phi_bcs(Phi)

        # select solver for velocity potential equation
        potential_solver = self._select_solver(
            self.solver_configs, "potential", "pressure_poisson"
        )
        Phi_needs_ref = needs_reference_value(Phi)

        n = self.potential_flow_config.n_non_orthogonal_correctors

        # solve velocity potential equation
        for _ in range(n + 1):
            phi_eqn_mat = fvm.laplacian(1.0, Phi)
            phi_eqn_mat.source = phi_eqn_mat.source + div_phi
            if Phi_needs_ref:
                set_reference_value(phi_eqn_mat)
            potential_eq = equation("potential", Phi, phi_eqn_mat)
            Phi.data = potential_solver.solve(potential_eq)

        # reconstruct velocity field
        sn_grad_Phi = fvc.sn_grad(Phi)
        mag_Sf = torch.linalg.vector_norm(
            Phi.grid.Sf[self.phi.single_mask], dim=1, keepdim=True
        )
        self.phi.single_data = (
            self.phi.single_data - sn_grad_Phi.single_data * mag_Sf
        )

        reconstruct_U_from_phi(self.phi, self.U)

        continuity_residual = torch.linalg.vector_norm(
            fvc.div(self.phi).data, ord=2
        ).item()
        logger.info(
            "potential flow corrected continuity residual L2=%.3e",
            continuity_residual,
        )

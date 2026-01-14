import pathlib

import torch

from gridfoam.CTX.context import SimulationContext
from gridfoam.DNA.ASTNodes._interface import IASTNode
from gridfoam.DNA.ASTNodes.arithmetic_node import ArithmeticNode
from gridfoam.DNA.ASTNodes.operator_node import OperatorNode
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.enum import OperatorType
from gridfoam.DNA.scheme.flux.correction import FluxCorrection
from gridfoam.DNA.scheme.fvm.div._linear import FVMDivLinear
from gridfoam.DNA.scheme.rhie_chow.correction import RhieChowCorrection
from gridfoam.RNA.grid_handle import GridHandle
from gridfoam.RNA.registry import SimulationMetaRegistry


class ICOEngine:
    def __init__(
        self,
        registry: SimulationMetaRegistry,
        configpath: pathlib.Path,
    ) -> None:
        grid_handle = GridHandle(configpath=configpath)
        config = grid_handle.config
        registry.register_scheme(config.simulator.fvSchemes)
        registry.register_solver(config.simulator.fvSolution)
        self._context = SimulationContext(
            registry=registry,
            grid_handle=grid_handle,
        )
        self._write_interval = config.simulator.control.writeInterval
        self._end_time = config.simulator.control.endTime
        self._deltaT = config.simulator.control.deltaT
        self._flux_correction = FluxCorrection(
            phi_fm=registry.get_field("phi"),
            U_fm=registry.get_field("U"),
            momentum_eq=registry.get_equation("momentum"),
        )
        self._rhie_chow_correction = RhieChowCorrection(
            U_fm=registry.get_field("U"),
            p_fm=registry.get_field("p"),
            momentum_eq=registry.get_equation("momentum"),
        )
        self._base_name = config.io.base_name

    def initialize(self) -> None:
        self.context.grid_handle.allocate_by_registry(self.context.registry)
        for equation_meta in self.context.registry.equations.values():
            self._initial_evaluation(equation_meta.ast_root)

    def _initial_evaluation(self, node: IASTNode) -> None:
        if isinstance(node, ArithmeticNode):
            self._initial_evaluation(node.arg1)
            self._initial_evaluation(node.arg2)
        if isinstance(node, OperatorNode):
            operator_type = node.type
            key = node.key
            registry = self.context.registry
            match operator_type:
                case OperatorType.DDT:
                    operator = registry.get_ddt_operator(key, node.args[0])
                case OperatorType.DIV:
                    n_args = len(node.args)
                    if n_args == 1:
                        operator = FVMDivLinear(node.args[0])
                    elif n_args == 2:
                        operator = registry.get_div_operator(
                            key, node.args[0], node.args[1]
                        )
                    else:
                        raise ValueError(
                            f"Unknown number of arguments: {n_args}"
                        )
                case OperatorType.GRAD:
                    operator = registry.get_grad_operator(key, node.args[0])
                case OperatorType.LAPLACIAN:
                    operator = registry.get_laplacian_operator(
                        key, node.args[0], node.args[1]
                    )
                case _:
                    raise ValueError(f"Unknown operator type: {operator_type}")
            node.operator = operator

    def solve(self) -> None:
        step = 0
        time = 0.0
        self.context.save(f"{self._base_name}_{step:04d}.vtkhdf")

        while 1:
            print(f"Step {step:04d}")
            step += 1
            time += self._deltaT

            # momentum equation
            momentum_eq = self.context.registry.equations["momentum"]
            self.context.grid_handle.update_fvmatrix(momentum_eq)
            solver = self.context.registry.get_solver(momentum_eq.name)
            solver.solve(self.context.grid_handle)

            # self.context.save(f"cavity_{step:04d}_momentum.vtkhdf")

            # momentum correction
            for level in self.context.grid_handle.iter_levels():
                dx = self.context.grid_handle.get_dx_at_depth(level.depth)
                for cube in level.nodes.values():
                    fvmatrix = cube.field.fvmatrices[momentum_eq.name]
                    ap = fvmatrix.a_P.interior
                    cube.field.cells["rAU"].interior = 1.0 / ap
            self.context.grid_handle.sync_all(fm_list=[self.context.registry.fields["rAU"]])

            # poisson equation
            poisson_eq = self.context.registry.equations["poisson"]
            self.context.grid_handle.update_fvmatrix(poisson_eq)
            solver = self.context.registry.get_solver(poisson_eq.name)
            solver.solve(self.context.grid_handle)

            # self.context.save(f"cavity_{step:04d}_poisson.vtkhdf")

            # rhie-chow correction
            for level in self.context.grid_handle.iter_levels():
                dx = self.context.grid_handle.get_dx_at_depth(level.depth)
                for cube in level.nodes.values():
                    ctx = CtxForCubeOperation(
                        depth=level.depth,
                        bounds=level.bounds,
                        dt=self.context.grid_handle.config.simulator.control.deltaT,
                        dx=dx,
                        vertices=torch.tensor(
                            self.context.grid_handle.mesh.points,
                            dtype=torch.float32,
                            device=self.context.grid_handle.config.cube.device,
                        ),
                        bcs=poisson_eq.boundary_conditions,
                    )
                    self._rhie_chow_correction.update_velocity(cube, ctx)
            self.context.grid_handle.sync_all(fm_list=[self.context.registry.fields["U"]])

            # self.context.save(f"cavity_{step:04d}_rhie_chow.vtkhdf")

            # flux correction
            for level in self.context.grid_handle.iter_levels():
                dx = self.context.grid_handle.get_dx_at_depth(level.depth)
                for cube in level.nodes.values():
                    ctx = CtxForCubeOperation(
                        depth=level.depth,
                        bounds=level.bounds,
                        dt=self.context.grid_handle.config.simulator.control.deltaT,
                        dx=dx,
                        vertices=torch.tensor(
                            self.context.grid_handle.mesh.points,
                            dtype=torch.float32,
                            device=self.context.grid_handle.config.cube.device,
                        ),
                        bcs=momentum_eq.boundary_conditions,
                    )
                    self._flux_correction.update_flux(cube, ctx)

            if step % self._write_interval == 0:
                self.context.save(
                    f"{self._base_name}_{step:04d}.vtkhdf"
                )
            if time >= self._end_time:
                break

    @property
    def context(self) -> SimulationContext:
        return self._context

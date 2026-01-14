import csv
import pathlib

from gridfoam.CTX.context import SimulationContext
from gridfoam.DNA.ASTNodes._interface import IASTNode
from gridfoam.DNA.ASTNodes.arithmetic_node import ArithmeticNode
from gridfoam.DNA.ASTNodes.operator_node import OperatorNode
from gridfoam.DNA.enum import OperatorType
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.scheme.fvm.div._linear import FVMDivLinear
from gridfoam.RNA.grid_handle import GridHandle
from gridfoam.RNA.registry import SimulationMetaRegistry


class SimulationEngine:
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

    def solve(self, equation_meta: EquationMeta) -> None:
        step = 0
        time = 0.0
        self.context.save(f"{self._base_name}_{step:04d}.vtkhdf")

        time_vs_total_T = {"time": [], "total_T": []}

        total_T = 0.0
        for _, cube in self.context.grid_handle.iter_all_leaves():
            field = cube.field
            T_field = field.cells["T"]
            cube_T = T_field.interior[0].sum()
            total_T += cube_T.item()
        time_vs_total_T["time"].append(time)
        time_vs_total_T["total_T"].append(total_T)

        while 1:
            print(f"Step {step:04d}")
            step += 1
            time += self._deltaT

            self.context.grid_handle.update_fvmatrix(equation_meta)
            solver = self.context.registry.get_solver(equation_meta.name)
            solver.solve(self.context.grid_handle)

            total_T = 0.0
            for _, cube in self.context.grid_handle.iter_all_leaves():
                field = cube.field
                T_field = field.cells["T"]
                cube_T = T_field.interior[0].sum()
                total_T += cube_T.item()
            time_vs_total_T["time"].append(time)
            time_vs_total_T["total_T"].append(total_T)

            if step % self._write_interval == 0:
                self.context.save(
                    f"{self._base_name}_{step:04d}.vtkhdf"
                )
            if time >= self._end_time:
                break

        with open("time_vs_total_T.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(time_vs_total_T.keys())
            writer.writerows(zip(*time_vs_total_T.values(), strict=True))

    @property
    def context(self) -> SimulationContext:
        return self._context

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
        self.context = SimulationContext(
            registry=registry,
            grid_handle=grid_handle,
        )

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
                    node.operator = operator
                case OperatorType.DIV:
                    n_args = len(node.args)
                    if n_args == 1:
                        operator = FVMDivLinear(node.args[0])
                    elif n_args == 2:
                        operator = registry.get_div_operator(
                            key, node.args[0], node.args[1]
                        )
                        node.operator = operator
                    else:
                        raise ValueError(
                            f"Unknown number of arguments: {n_args}"
                        )
                case OperatorType.GRAD:
                    operator = registry.get_grad_operator(key, node.args[0])
                    node.operator = operator
                case OperatorType.LAPLACIAN:
                    operator = registry.get_laplacian_operator(
                        key, node.args[0], node.args[1]
                    )
                    node.operator = operator
                case _:
                    raise ValueError(f"Unknown operator type: {operator_type}")

    def solve(self, equation_meta: EquationMeta) -> None:
        self.context.grid_handle.update_fvmatrix(equation_meta)
        solver = self.context.registry.get_solver(equation_meta.name)
        solver.solve(self.context.grid_handle)

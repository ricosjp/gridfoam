from typing import Literal

from gridfoam._base._equation._expr import Expr
from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._interface._fvm_term import IFVMTerm
from gridfoam._base._schemes._ddt import match_choice_to_ddt_scheme
from gridfoam._base._schemes._div import match_choice_to_div_scheme
from gridfoam._base._schemes._laplacian import match_choice_to_laplacian_scheme
from gridfoam.config import fvSchemesConfig


class SchemeRegistry:
    def __init__(self, config: fvSchemesConfig):
        self._scheme_config = config

    def _get_ddt_scheme(self, target_fd: FieldDescriptor) -> IFVMTerm:
        ddtSchemes = self._scheme_config.ddtSchemes
        assert len(ddtSchemes) == 1
        assert "default" in ddtSchemes
        scheme_choice = ddtSchemes["default"]
        return match_choice_to_ddt_scheme(scheme_choice, target_fd)

    def _get_div_scheme(
        self, velocity_fd: FieldDescriptor, target_fd: FieldDescriptor
    ) -> IFVMTerm:
        divSchemes = self._scheme_config.divSchemes
        if divSchemes is None:
            raise ValueError("div scheme is not registered")
        key = f"div({velocity_fd.alias}, {target_fd.alias})"
        scheme_choice = divSchemes.get(key)
        if scheme_choice is None:
            assert "default" in divSchemes
            scheme_choice = divSchemes["default"]
        return match_choice_to_div_scheme(scheme_choice, velocity_fd, target_fd)

    def _get_laplacian_scheme(
        self, gamma_fd: FieldDescriptor, target_fd: FieldDescriptor
    ) -> IFVMTerm:
        laplacianSchemes = self._scheme_config.laplacianSchemes
        if laplacianSchemes is None:
            raise ValueError("laplacian scheme is not registered")
        key = f"laplacian({gamma_fd.alias}, {target_fd.alias})"
        scheme_choice = laplacianSchemes.get(key)
        if scheme_choice is None:
            assert "default" in laplacianSchemes
            scheme_choice = laplacianSchemes["default"]
        return match_choice_to_laplacian_scheme(
            scheme_choice, gamma_fd, target_fd
        )

    def _get_scheme(
        self,
        op_name: Literal["ddt", "div", "laplacian"],
        args: list[FieldDescriptor],
    ) -> IFVMTerm:
        match op_name:
            case "ddt":
                return self._get_ddt_scheme(args[0])
            case "div":
                return self._get_div_scheme(args[0], args[1])
            case "laplacian":
                return self._get_laplacian_scheme(args[0], args[1])
            case _:
                raise ValueError(f"Unknown operator: {op_name}")

    def resolve(self, expr: Expr) -> IFVMTerm:
        op_name = expr.op_name
        match op_name:
            case "Add":
                return self.resolve(expr.args[0]) + self.resolve(expr.args[1])
            case "Sub":
                return self.resolve(expr.args[0]) - self.resolve(expr.args[1])
            case "ddt" | "div" | "laplacian":
                args = expr.args
                return self._get_scheme(op_name, args)
            case _:
                raise ValueError(f"Unknown operator: {op_name}")

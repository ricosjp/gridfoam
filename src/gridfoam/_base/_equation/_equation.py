from __future__ import annotations

from dataclasses import dataclass

from gridfoam._base._equation._expr import Expr


@dataclass
class Equation:
    expr: Expr
    tag: str

    def __repr__(self) -> str:
        return f"Equation(expr={self.expr}, tag={self.tag})"

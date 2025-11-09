from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Expr:
    op_name: str
    args: list

    def __add__(self, other: Expr) -> Expr:
        return Expr(op_name="Add", args=[self, other])

    def __sub__(self, other: Expr) -> Expr:
        return Expr(op_name="Sub", args=[self, other])

    # def __mul__(self, other: Expr) -> Expr:
    #     return Expr(op_name="Mult", args=[self, other])

    # def __rmul__(self, other: float) -> Expr:
    #     return Expr(op_name="ScalarMult", args=[other, self])

    # def __repr__(self) -> str:
    #     match self.op_name:
    #         case "Add":
    #             return f"{self.args[0]} + {self.args[1]}"
    #         case "Sub":
    #             return f"{self.args[0]} - {self.args[1]}"
    #         case "Mult":
    #             return f"{self.args[0]} * {self.args[1]}"
    #         case "ScalarMult":
    #             return f"{self.args[0]} * {self.args[1]}"
    #         case _:
    #             return f"{self.op_name}({', '.join(map(str, self.args))})"

    # @property
    # def key(self) -> str:
    #     return str(self)

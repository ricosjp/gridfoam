import abc

from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.fielddata import FVMatrix


class IFVMOperator(abc.ABC):
    @abc.abstractmethod
    def build(
        self,
        cube_field: CubeField,
    ) -> FVMatrix:
        """
        Build the FVM operator.
        """
        pass

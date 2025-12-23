import abc

from gridfoam.DNA._grid._grid import PyOctreeNode
from gridfoam.DNA.fielddata import FVMatrix


class IFVMOperator(abc.ABC):
    @abc.abstractmethod
    def build(
        self,
        cube: PyOctreeNode,
    ) -> FVMatrix:
        """
        Build the FVM operator.
        """
        pass

import abc

from gridfoam.DNA._grid._grid import PyOctreeNode
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.fielddata import FVMatrix


class IFVMOperator(abc.ABC):
    @abc.abstractmethod
    def build(
        self,
        cube: PyOctreeNode,
        ctx: CtxForCubeOperation,
    ) -> FVMatrix:
        """
        Build the FVM operator.
        """
        pass

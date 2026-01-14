import abc

from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm._interface import IFVMOperator


class IFVMDdtOperator(IFVMOperator):
    @abc.abstractmethod
    def __init__(self, psi_fm: FieldMeta) -> None:
        pass

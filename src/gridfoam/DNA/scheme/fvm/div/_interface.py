import abc

from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm._interface import IFVMOperator


class IFVMDivOperator(IFVMOperator):
    @abc.abstractmethod
    def __init__(self, phi_fm: FieldMeta, psi_fm: FieldMeta) -> None:
        pass

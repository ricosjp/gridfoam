import abc


class IFVCOperator(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def apply(
        cls,
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        Apply the FVC operator.
        """
        pass


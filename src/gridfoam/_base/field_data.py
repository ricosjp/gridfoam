from collections.abc import ItemsView

import torch


class FieldData:
    """
    Container for field data tensors.

    Parameters
    ----------
    device : torch.device
        The device on which tensors are allocated.

    Notes
    -----
    Fields can be accessed as attributes or via dictionary-style access.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._field_data: dict[str, torch.Tensor] = {}

    def add_field(self, name: str, shape: tuple, dtype: torch.dtype) -> None:
        """
        Add a new field tensor.

        Parameters
        ----------
        name : str
            Field name.
        shape : tuple
            Shape of the tensor.
        dtype : torch.dtype
            Data type of the tensor.
        """
        data = torch.zeros(shape, dtype=dtype, device=self.device)
        self._field_data[name] = data

    def __getitem__(self, name: str) -> torch.Tensor:
        """
        Get a field tensor by name.

        Parameters
        ----------
        name : str
            Field name.

        Returns
        -------
        torch.Tensor
            The corresponding tensor.
        """
        if name in self._field_data:
            return self._field_data[name]
        raise AttributeError(f"'FieldData' object has no attribute '{name}'")

    def __getattr__(self, name: str) -> torch.Tensor:
        """
        Allow attribute-style access to fields.

        Parameters
        ----------
        name : str
            Field name.

        Returns
        -------
        torch.Tensor
            The corresponding tensor.

        Raises
        ------
        AttributeError
            If the field does not exist.
        """
        if name in self._field_data:
            return self._field_data[name]
        raise AttributeError(f"'FieldData' object has no attribute '{name}'")

    def items(self) -> ItemsView[str, torch.Tensor]:
        """
        Return an iterator over (field name, tensor) pairs.

        Returns
        -------
        Iterator[tuple[str, torch.Tensor]]
            Iterator over field data items.
        """
        return self._field_data.items()

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch


class BaseEstimator(ABC):
    """Base class for all estimators in torchonometrics.

    Parameters
    ----------
    device : torch.device, str, or None, default=None
        Device to use for computations. If None, automatically selects
        'cuda' if available, otherwise 'cpu'.
    """

    def __init__(self, device: Optional[Union[torch.device, str]] = None):
        # Auto-detect best device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
        self.params: Optional[Dict[str, torch.Tensor]] = None

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseEstimator":
        """Fit the model to the data."""
        raise NotImplementedError

    def to(self, device: Union[torch.device, str]) -> "BaseEstimator":
        """Move all model parameters to specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device ('cuda', 'cpu', etc.)

        Returns
        -------
        BaseEstimator
            Self for method chaining
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.params:
            self.params = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in self.params.items()
            }
        return self

    def summary(self) -> None:
        """Print a summary of the model results."""
        if self.params is None:
            print("Model has not been fitted yet.")
            return

        print(f"{self.__class__.__name__} Results")
        print("=" * 30)
        for param_name, param_value in self.params.items():
            print(f"{param_name}: {param_value}")
        print("=" * 30)

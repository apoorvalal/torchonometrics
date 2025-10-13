from abc import abstractmethod
import torch

from torchonometrics.mle import MaximumLikelihoodEstimator


class ChoiceModel(MaximumLikelihoodEstimator):
    """
    Base class for structural choice models.
    """

    @abstractmethod
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict choice probabilities.
        """
        raise NotImplementedError

    @abstractmethod
    def simulate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simulate choices.
        """
        raise NotImplementedError

    @abstractmethod
    def counterfactual(self, X_new: torch.Tensor) -> dict:
        """
        Perform counterfactual analysis.
        """
        raise NotImplementedError

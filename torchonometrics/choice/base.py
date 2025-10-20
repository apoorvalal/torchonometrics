from abc import abstractmethod
import torch

from torchonometrics.mle import MaximumLikelihoodEstimator


class ChoiceModel(MaximumLikelihoodEstimator):
    """
    Base class for structural discrete choice models.

    This abstract class provides the foundation for implementing utility-based
    choice models such as logit, probit, and multinomial logit. All subclasses
    must implement methods for predicting choice probabilities, simulating choices,
    and performing counterfactual analysis.

    Inherits from MaximumLikelihoodEstimator, which provides:
        - fit() method for parameter estimation via optimization
        - Standard error computation via Fisher information
        - Model summary statistics

    Subclasses must implement:
        - _negative_log_likelihood(): Model-specific likelihood function
        - _compute_fisher_information(): Fisher information matrix computation
        - predict_proba(): Predict choice probabilities
        - simulate(): Simulate choices from the model
        - counterfactual(): Perform policy/counterfactual analysis
    """

    @abstractmethod
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict choice probabilities for given covariates.

        Args:
            X: Design matrix or feature tensor. Shape depends on model type.

        Returns:
            Predicted choice probabilities. Shape depends on model type:
                - Binary models: (n_samples,)
                - Multinomial models: (n_samples, n_choices)

        Raises:
            ValueError: If model has not been fitted yet.
        """
        raise NotImplementedError

    @abstractmethod
    def simulate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simulate choices from the fitted model given covariates.

        Uses the fitted parameters to generate synthetic choice data by
        sampling from the predicted choice probabilities.

        Args:
            X: Design matrix or feature tensor. Shape depends on model type.

        Returns:
            Simulated choices. Shape and type depend on model:
                - Binary models: (n_samples,) with values in {0, 1}
                - Multinomial models: (n_samples,) with choice indices

        Raises:
            ValueError: If model has not been fitted yet.
        """
        raise NotImplementedError

    @abstractmethod
    def counterfactual(self, X_new: torch.Tensor) -> dict:
        """
        Perform counterfactual analysis comparing scenarios.

        Evaluates how choice probabilities and market shares change under
        counterfactual scenarios (e.g., price changes, product entry/exit,
        policy interventions).

        Args:
            X_new: Counterfactual design matrix representing the alternative scenario.

        Returns:
            Dictionary containing counterfactual results. Common keys include:
                - market_share_original: Baseline market shares
                - market_share_counterfactual: Counterfactual market shares
                - change_in_market_share: Difference in market shares

        Raises:
            ValueError: If model has not been fitted yet.
        """
        raise NotImplementedError

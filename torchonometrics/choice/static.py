import torch
from scipy import stats

from .base import ChoiceModel


class BinaryLogit(ChoiceModel):
    """
    Binary Logit model for structural estimation.
    """

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        logits = X @ params
        return -torch.sum(
            y * torch.nn.functional.logsigmoid(logits)
            + (1 - y) * torch.nn.functional.logsigmoid(-logits)
        )

    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        logits = X @ params
        probs = torch.sigmoid(logits)
        weights = probs * (1 - probs)
        weighted_X = X * weights.unsqueeze(1)
        return weighted_X.T @ X

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")
        logits = X @ self.params["coef"]
        return torch.sigmoid(logits)

    def simulate(self, X: torch.Tensor) -> torch.Tensor:
        probas = self.predict_proba(X)
        return (torch.rand_like(probas) < probas).to(torch.int32)

    def counterfactual(self, X_new: torch.Tensor) -> dict:
        """
        Computes the change in market share for a counterfactual scenario.
        """
        if self._fitted_X is None:
            raise ValueError("Model must be fitted before running counterfactuals.")

        # Original market share
        p_old = self.predict_proba(self._fitted_X)
        market_share_old = torch.mean(p_old)

        # Counterfactual market share
        p_new = self.predict_proba(X_new)
        market_share_new = torch.mean(p_new)

        return {
            "market_share_original": market_share_old,
            "market_share_counterfactual": market_share_new,
            "change_in_market_share": market_share_new - market_share_old,
        }


class BinaryProbit(ChoiceModel):
    """
    Binary Probit model for structural estimation.
    """

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        logits = X @ params
        norm_dist = torch.distributions.Normal(0, 1)
        # Add a small epsilon for numerical stability
        eps = 1e-8
        log_likelihood = torch.sum(
            y * torch.log(norm_dist.cdf(logits) + eps)
            + (1 - y) * torch.log(norm_dist.cdf(-logits) + eps)
        )
        return -log_likelihood

    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        logits = X @ params
        norm_dist = torch.distributions.Normal(0, 1)
        pdf_vals = torch.exp(norm_dist.log_prob(logits))
        cdf_vals = norm_dist.cdf(logits)
        weights = pdf_vals**2 / (cdf_vals * (1 - cdf_vals))
        weighted_X = X * weights.unsqueeze(1)
        return weighted_X.T @ X

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")
        logits = X @ self.params["coef"]
        norm_dist = torch.distributions.Normal(0, 1)
        return norm_dist.cdf(logits)

    def simulate(self, X: torch.Tensor) -> torch.Tensor:
        probas = self.predict_proba(X)
        return (torch.rand_like(probas) < probas).to(torch.int32)

    def counterfactual(self, X_new: torch.Tensor) -> dict:
        """
        Computes the change in market share for a counterfactual scenario.
        """
        if self._fitted_X is None:
            raise ValueError("Model must be fitted before running counterfactuals.")

        # Original market share
        p_old = self.predict_proba(self._fitted_X)
        market_share_old = torch.mean(p_old)

        # Counterfactual market share
        p_new = self.predict_proba(X_new)
        market_share_new = torch.mean(p_new)

        return {
            "market_share_original": market_share_old,
            "market_share_counterfactual": market_share_new,
            "change_in_market_share": market_share_new - market_share_old,
        }


class MultinomialLogit(ChoiceModel):
    """
    Multinomial Logit model for structural estimation.
    """

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        n_choices = y.shape[1]
        # params are shaped (n_features, n_choices - 1)
        # We fix one choice's params to 0 for identification
        params_full = torch.cat([params, torch.zeros(X.shape[1], 1)], dim=1)
        logits = X @ params_full
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        return -torch.sum(y * log_probs)

    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # This is more complex for multinomial logit and will be implemented later.
        return torch.eye(params.shape[0])

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")
        params_full = torch.cat(
            [self.params["coef"], torch.zeros(X.shape[1], 1)], dim=1
        )
        logits = X @ params_full
        return torch.nn.functional.softmax(logits, dim=1)

    def simulate(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(X)
        return torch.multinomial(probs, 1).squeeze(1)

    def counterfactual(self, X_new: torch.Tensor) -> dict:
        if self._fitted_X is None:
            raise ValueError("Model must be fitted before running counterfactuals.")

        # Original market share
        p_old = self.predict_proba(self._fitted_X)
        market_share_old = torch.mean(p_old, dim=0)

        # Counterfactual market share
        p_new = self.predict_proba(X_new)
        market_share_new = torch.mean(p_new, dim=0)

        return {
            "market_share_original": market_share_old,
            "market_share_counterfactual": market_share_new,
            "change_in_market_share": market_share_new - market_share_old,
        }


class NestedLogit(ChoiceModel):
    """
    Nested Logit model for structural estimation.
    """

    def __init__(self, nesting_structure: dict, optimizer=None, maxiter=5000, tol=1e-4):
        super().__init__(optimizer, maxiter, tol)
        self.nesting_structure = nesting_structure

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        # This is a simplified placeholder. A full implementation would need to unpack
        # params into coefficients and lambda values.
        probs = self.predict_proba(X)
        return -torch.sum(y * torch.log(probs + 1e-8))

    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # Implementation will be complex and will be done in a subsequent step.
        return torch.eye(params.shape[0])

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if not self.params or "coef" not in self.params or "lambda" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        n_samples = X.shape[0]
        n_choices = sum(len(v) for v in self.nesting_structure.values())
        probs = torch.zeros(n_samples, n_choices)

        inclusive_values = {
            nest: torch.zeros(n_samples) for nest in self.nesting_structure
        }

        # Calculate inclusive values for each nest
        for nest, choices in self.nesting_structure.items():
            for choice in choices:
                inclusive_values[nest] += torch.exp(
                    X @ self.params["coef"][:, choice] / self.params["lambda"][nest]
                )

        # Calculate nest probabilities
        nest_probs = torch.zeros(n_samples, len(self.nesting_structure))
        total_inclusive_value = torch.zeros(n_samples)
        for i, (nest, iv) in enumerate(inclusive_values.items()):
            total_inclusive_value += torch.pow(iv, self.params["lambda"][nest])

        for i, (nest, iv) in enumerate(inclusive_values.items()):
            nest_probs[:, i] = (
                torch.pow(iv, self.params["lambda"][nest]) / total_inclusive_value
            )

        # Calculate final choice probabilities
        for i, (nest, choices) in enumerate(self.nesting_structure.items()):
            for choice in choices:
                within_nest_prob = (
                    torch.exp(
                        X @ self.params["coef"][:, choice] / self.params["lambda"][nest]
                    )
                    / inclusive_values[nest]
                )
                probs[:, choice] = nest_probs[:, i] * within_nest_prob

        return probs

    def simulate(self, X: torch.Tensor) -> torch.Tensor:
        # Implementation will be complex and will be done in a subsequent step.
        pass

    def counterfactual(self, X_new: torch.Tensor) -> dict:
        # Implementation will be complex and will be done in a subsequent step.
        pass

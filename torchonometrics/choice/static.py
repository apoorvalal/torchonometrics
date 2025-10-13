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

    def _compute_fisher_information(self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
            y * torch.log(norm_dist.cdf(logits) + eps) + (1 - y) * torch.log(norm_dist.cdf(-logits) + eps)
        )
        return -log_likelihood

    def _compute_fisher_information(self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    def _compute_fisher_information(self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # This is more complex for multinomial logit and will be implemented later.
        return torch.eye(params.shape[0])

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")
        params_full = torch.cat([self.params["coef"], torch.zeros(X.shape[1], 1)], dim=1)
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


class LowRankLogit(ChoiceModel):
    """
    Low-Rank Logit model for matrix completion style choice problems.
    """

    def __init__(self, rank: int, n_users: int, n_items: int, optimizer=torch.optim.LBFGS, maxiter=20000, tol=1e-4):
        super().__init__(optimizer, maxiter, tol)
        self.rank = rank
        self.n_users = n_users
        self.n_items = n_items

    def _negative_log_likelihood(self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> float:
        # X is expected to be a tensor of user indices
        # y is expected to be a tensor of chosen item indices
        user_indices = X.long()
        item_indices = y.long()

        A = params[:self.n_users * self.rank].reshape(self.n_users, self.rank)
        B = params[self.n_users * self.rank:].reshape(self.n_items, self.rank)

        theta = A @ B.T
        
        # For each observation, we need to compute the log-softmax over the choice set.
        # This is a simplified version assuming the choice set is all items.
        # A full implementation would need to handle varying choice sets.
        log_probs = torch.nn.functional.log_softmax(theta, dim=1)
        nll = -torch.sum(log_probs[user_indices, item_indices])

        return nll

    def fit(self, X: torch.Tensor, y: torch.Tensor, init_params: torch.Tensor = None, verbose: bool = False) -> "LowRankLogit":
        if init_params is None:
            A_init = torch.randn(self.n_users, self.rank) * 0.1
            B_init = torch.randn(self.n_items, self.rank) * 0.1
            init_params = torch.cat([A_init.flatten(), B_init.flatten()])

        current_params = init_params.clone().requires_grad_(True)

        if self.optimizer_class == torch.optim.LBFGS:
            optimizer = self.optimizer_class([current_params], max_iter=20)
        else:
            optimizer = self.optimizer_class([current_params])

        self.history["loss"] = []

        for i in range(self.maxiter):
            def closure():
                optimizer.zero_grad()
                loss = self._negative_log_likelihood(current_params, X, y)
                loss.backward()
                return loss

            if self.optimizer_class == torch.optim.LBFGS:
                loss_val = optimizer.step(closure)
            else:
                loss_val = closure()
                optimizer.step()

            self.history["loss"].append(loss_val.item())

            if i > 10 and self.tol > 0:
                loss_change = abs(self.history["loss"][-2] - self.history["loss"][-1]) / (abs(self.history["loss"][-2]) + 1e-8)
                if loss_change < self.tol:
                    if verbose:
                        print(f"Convergence tolerance {self.tol} met at iteration {i}.")
                    break

        self.params = {"coef": current_params.detach()}
        self.iterations_run = i + 1
        self._fitted_X = X.detach()
        self._fitted_y = y.detach()
        self._compute_standard_errors()

        # Unpack and store final parameters
        self.params["A"] = self.params["coef"][:self.n_users * self.rank].reshape(self.n_users, self.rank)
        self.params["B"] = self.params["coef"][self.n_users * self.rank:].reshape(self.n_items, self.rank)
        return self

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if not self.params or "A" not in self.params or "B" not in self.params:
            raise ValueError("Model has not been fitted yet.")
        
        theta = self.params["A"] @ self.params["B"].T
        return torch.nn.functional.softmax(theta, dim=1)

    def _compute_fisher_information(self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # This is a placeholder and should be implemented properly.
        return torch.eye(params.shape[0])

    def simulate(self, X: torch.Tensor) -> torch.Tensor:
        # X is expected to be a tensor of user indices
        user_indices = X.long()
        probs = self.predict_proba(user_indices)
        return torch.multinomial(probs, 1).squeeze(1)

    def counterfactual(self, X_new: torch.Tensor) -> dict:
        # This is a placeholder and should be implemented properly.
        return {}

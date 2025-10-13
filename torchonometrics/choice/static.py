import torch

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


class LowRankLogit(ChoiceModel):
    """
    Low-Rank Logit model for matrix completion style choice problems.

    This model handles varying choice sets (assortments) by learning a low-rank
    factorization Θ = AB^T of the user-item utility matrix, where users choose
    from presented assortments according to multinomial logit probabilities.

    Based on: Kallus & Udell (2016), "Revealed Preference at Scale: Learning
    Personalized Preferences from Assortment Choices"
    """

    def __init__(
        self,
        rank: int,
        n_users: int,
        n_items: int,
        lam: float = 0.0,
        optimizer=torch.optim.LBFGS,
        maxiter=20000,
        tol=1e-4,
    ):
        super().__init__(optimizer, maxiter, tol)
        self.rank = rank
        self.n_users = n_users
        self.n_items = n_items
        self.lam = lam  # Regularization parameter

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        assortments: torch.Tensor = None,
    ) -> float:
        """
        Compute negative log-likelihood for observations with varying choice sets.

        Args:
            params: Flattened parameters [A; B] where A is n_users x rank, B is n_items x rank
            X: User indices (n_obs,)
            y: Chosen item indices (n_obs,)
            assortments: Binary mask (n_obs, n_items) indicating available items per observation.
                        If None, assumes all items are available (full choice set).

        Returns:
            Negative log-likelihood
        """
        user_indices = X.long()
        item_indices = y.long()

        A = params[: self.n_users * self.rank].reshape(self.n_users, self.rank)
        B = params[self.n_users * self.rank :].reshape(self.n_items, self.rank)

        theta = A @ B.T

        # Enforce zero-sum constraint (Θe = 0) per user for identification
        theta = theta - theta.mean(dim=1, keepdim=True)

        # Select utilities for observed users
        utilities = theta[user_indices]  # (n_obs, n_items)

        # Mask unavailable items by setting their utilities to -inf
        if assortments is not None:
            utilities = torch.where(
                assortments.bool(), utilities, torch.tensor(float("-inf"))
            )

        # Compute log-softmax over available items
        log_probs = torch.nn.functional.log_softmax(utilities, dim=1)
        nll = -torch.sum(log_probs[torch.arange(len(item_indices)), item_indices])

        # Add regularization: λ/2(||A||_F^2 + ||B||_F^2) approximates λ||Θ||_*
        if self.lam > 0:
            reg = 0.5 * self.lam * (torch.sum(A**2) + torch.sum(B**2))
            return nll + reg

        return nll

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        assortments: torch.Tensor = None,
        init_params: torch.Tensor = None,
        verbose: bool = False,
    ) -> "LowRankLogit":
        """
        Fit the low-rank logit model.

        Args:
            X: User indices (n_obs,)
            y: Chosen item indices (n_obs,)
            assortments: Binary mask (n_obs, n_items) where 1 indicates item was available.
                        If None, assumes all items are always available.
            init_params: Initial parameter values (optional)
            verbose: Print convergence information

        Returns:
            self
        """
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
                loss = self._negative_log_likelihood(current_params, X, y, assortments)
                loss.backward()
                return loss

            if self.optimizer_class == torch.optim.LBFGS:
                loss_val = optimizer.step(closure)
            else:
                loss_val = closure()
                optimizer.step()

            self.history["loss"].append(loss_val.item())

            if i > 10 and self.tol > 0:
                loss_change = abs(
                    self.history["loss"][-2] - self.history["loss"][-1]
                ) / (abs(self.history["loss"][-2]) + 1e-8)
                if loss_change < self.tol:
                    if verbose:
                        print(f"Convergence tolerance {self.tol} met at iteration {i}.")
                    break

        self.params = {"coef": current_params.detach()}
        self.iterations_run = i + 1
        self._fitted_X = X.detach()
        self._fitted_y = y.detach()
        self._compute_standard_errors()

        # Unpack and store final parameters with zero-sum constraint
        A = self.params["coef"][: self.n_users * self.rank].reshape(
            self.n_users, self.rank
        )
        B = self.params["coef"][self.n_users * self.rank :].reshape(
            self.n_items, self.rank
        )
        theta = A @ B.T
        theta = theta - theta.mean(dim=1, keepdim=True)

        self.params["A"] = A
        self.params["B"] = B
        self.params["theta"] = theta
        return self

    def predict_proba(
        self, X: torch.Tensor, assortments: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Predict choice probabilities.

        Args:
            X: User indices (n_obs,)
            assortments: Binary mask (n_obs, n_items) indicating available items.
                        If None, assumes all items are available.

        Returns:
            Choice probabilities (n_obs, n_items)
        """
        if not self.params or "theta" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        user_indices = X.long()
        utilities = self.params["theta"][user_indices]

        # Mask unavailable items
        if assortments is not None:
            utilities = torch.where(
                assortments.bool(), utilities, torch.tensor(float("-inf"))
            )

        return torch.nn.functional.softmax(utilities, dim=1)

    def _compute_fisher_information(
        self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # This is a placeholder and should be implemented properly.
        return torch.eye(params.shape[0])

    def simulate(
        self, X: torch.Tensor, assortments: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Simulate choices from the fitted model.

        Args:
            X: User indices (n_obs,)
            assortments: Binary mask (n_obs, n_items) indicating available items.
                        If None, assumes all items are available.

        Returns:
            Simulated item choices (n_obs,)
        """
        probs = self.predict_proba(X, assortments)
        return torch.multinomial(probs, 1).squeeze(1)

    def counterfactual(
        self,
        user_indices: torch.Tensor,
        baseline_assortments: torch.Tensor,
        counterfactual_assortments: torch.Tensor,
        item_revenues: torch.Tensor = None,
    ) -> dict:
        """
        Perform counterfactual analysis comparing two assortment scenarios.

        This method evaluates the impact of changing the choice set (assortment)
        on market shares, choice probabilities, and optionally expected revenues.
        The low-rank structure allows heterogeneous user preferences, avoiding
        the IIA problem of standard multinomial logit.

        Args:
            user_indices: User indices to evaluate (n_obs,)
            baseline_assortments: Binary mask (n_obs, n_items) for baseline scenario
            counterfactual_assortments: Binary mask (n_obs, n_items) for counterfactual
            item_revenues: Optional revenue per item (n_items,). If provided, computes
                          expected revenues under each scenario.

        Returns:
            dict containing:
                - baseline_probs: Choice probabilities under baseline (n_obs, n_items)
                - counterfactual_probs: Choice probabilities under counterfactual
                - baseline_market_share: Average choice probability per item (n_items,)
                - counterfactual_market_share: Average choice probability per item
                - market_share_change: Change in market share (n_items,)
                - baseline_expected_revenue: Expected revenue under baseline (if revenues provided)
                - counterfactual_expected_revenue: Expected revenue under counterfactual
                - revenue_change: Change in expected revenue
                - baseline_choices: Most likely choice per user under baseline (n_obs,)
                - counterfactual_choices: Most likely choice per user under counterfactual

        Example:
            >>> # Evaluate impact of adding a new product (item 5) to assortments
            >>> baseline = torch.ones(n_obs, n_items)
            >>> baseline[:, 5] = 0  # Product 5 not available in baseline
            >>> counterfactual = torch.ones(n_obs, n_items)  # All products available
            >>> results = model.counterfactual(users, baseline, counterfactual)
            >>> print(f"Adding product 5 changes market share by {results['market_share_change'][5]:.2%}")
        """
        if not self.params or "theta" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        # Compute choice probabilities under both scenarios
        baseline_probs = self.predict_proba(user_indices, baseline_assortments)
        counterfactual_probs = self.predict_proba(
            user_indices, counterfactual_assortments
        )

        # Compute market shares (average probability of choosing each item)
        baseline_market_share = baseline_probs.mean(dim=0)
        counterfactual_market_share = counterfactual_probs.mean(dim=0)
        market_share_change = counterfactual_market_share - baseline_market_share

        # Most likely choice per user (argmax)
        baseline_choices = torch.argmax(baseline_probs, dim=1)
        counterfactual_choices = torch.argmax(counterfactual_probs, dim=1)

        results = {
            "baseline_probs": baseline_probs,
            "counterfactual_probs": counterfactual_probs,
            "baseline_market_share": baseline_market_share,
            "counterfactual_market_share": counterfactual_market_share,
            "market_share_change": market_share_change,
            "baseline_choices": baseline_choices,
            "counterfactual_choices": counterfactual_choices,
        }

        # Compute expected revenues if provided
        if item_revenues is not None:
            # Expected revenue = sum over items of (choice probability * revenue)
            baseline_revenue = torch.sum(baseline_probs * item_revenues)
            counterfactual_revenue = torch.sum(counterfactual_probs * item_revenues)

            results["baseline_expected_revenue"] = baseline_revenue
            results["counterfactual_expected_revenue"] = counterfactual_revenue
            results["revenue_change"] = counterfactual_revenue - baseline_revenue
            results["revenue_change_pct"] = (
                (counterfactual_revenue - baseline_revenue) / baseline_revenue
            )

        return results

import numpy as np
from scipy.optimize import minimize

class LogisticRegression:
    """
    Regularised logistic regression with batch updates.
    Implementation according to Chapelle and Li (2011)
    """
    def __init__(self, n_features, lambda_reg):
        self.m = np.random.uniform(-0.01, 0.01, n_features)  # Small random values
        self.q = np.ones(n_features) * lambda_reg  # Precision (inverse variance)

    def sample_weights(self, alpha):
        """
        Sample weights from Gaussian posterior for Thompson Sampling.
        """
        return np.random.normal(self.m, np.sqrt(1 / self.q) * alpha)

    def update_posterior(self, X, y):
        """
        Updates the posterior distribution using a batch of training data.
        :param X: Feature matrix.
        :param y: Labels (-1 or +1).
        """
        def objective(w):
            reg_term = 0.5 * np.sum(self.q * (w - self.m) ** 2)
            log_loss = np.sum(np.log(1 + np.exp(-y * (X @ w))))
            return reg_term + log_loss

        result = minimize(objective, self.m, method='L-BFGS-B')
        self.m = result.x
        predictions = 1 / (1 + np.exp(-X @ self.m))
        self.q += np.sum(X ** 2 * predictions * (1 - predictions), axis=0)

class ContextualBandit:
    """
    Contextual bandit framework supporting multiple policies.
    """
    def __init__(self, n_features, n_actions, lambda_reg, strategy="thompson", alpha_thompson=1.0, alpha_linucb=1.0, epsilon=0.1):
        self.n_actions = n_actions
        self.strategy = strategy.lower()
        self.epsilon = epsilon  # Epsilon value for epsilon-greedy strategy

        # Create LogisticRegression models for each action
        self.models = [LogisticRegression(n_features, lambda_reg) for _ in range(n_actions)]

        # Alpha values specific to strategies
        self.alpha_thompson = alpha_thompson
        self.alpha_linucb = alpha_linucb

    def select_action(self, X):
        """
        Select action based on the chosen strategy.
        :param X: Context feature vector.
        :return: Selected action index.
        """
        if self.strategy == "thompson":
            sampled_rewards = [np.dot(model.sample_weights(self.alpha_thompson), X) for model in self.models]
            return np.argmax(sampled_rewards)

        elif self.strategy == "linucb":
            estimates = [
                model.m @ X + self.alpha_linucb * np.sqrt(np.sum(1 / model.q * X**2))
                for model in self.models
            ]
            return np.argmax(estimates)

        elif self.strategy == "exploit":
            estimated_means = [model.m @ X for model in self.models]
            return np.argmax(estimated_means)

        elif self.strategy == "epsilon-greedy":
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_actions)  # Explore
            else:
                estimated_means = [model.m @ X for model in self.models]
                return np.argmax(estimated_means)  # Exploit

        elif self.strategy == "random":
            return np.random.randint(self.n_actions)

        else:
            raise ValueError("Invalid strategy selected. Choose from ['thompson', 'linucb', 'exploit', 'random']")

    def batch_update(self, X_batch, actions, rewards):
        """
        Perform batch updates using multiple observations at once.
        :param X_batch: Batch of feature vectors.
        :param actions: Batch of actions taken.
        :param rewards: Batch of observed rewards (0 or 1).
        """
        for i in range(len(actions)):
            y = 2 * rewards[i] - 1  # Convert rewards to {-1, +1}
            self.models[actions[i]].update_posterior(X_batch[i].reshape(1, -1), np.array([y]))

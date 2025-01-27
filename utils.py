import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_data(n_samples, n_features, n_actions, with_noise=False):
    """
    Generates synthetic contextual bandit data with an option to add noise.

    :param n_samples: Number of data samples.
    :param n_features: Number of contextual features.
    :param n_actions: Number of possible actions.
    :param with_noise: Boolean flag to add noise to rewards.
    :return: Feature matrix X and reward probabilities.
    """
    np.random.seed(42)

    # Generate random feature matrix
    X = np.random.randn(n_samples, n_features)

    # True weight matrix for each action
    true_weights = np.random.randn(n_actions, n_features)

    # Compute reward probabilities
    rewards = X @ true_weights.T  # Linear combination of features and weights

    if with_noise:
        noise = np.random.normal(0, 5.0, size=(n_samples, n_actions))  # Add Gaussian noise
        rewards += noise

    rewards = 1 / (1 + np.exp(-rewards))  # Apply sigmoid for probability scaling

    return X, rewards

def plot_results(strategy_names, rewards, cumulative_regrets):
    """
    Plots the total rewards for each bandit strategy and the cumulative regret over time.

    :param strategy_names: List of strategy names.
    :param rewards: Corresponding rewards obtained for each strategy.
    :param cumulative_regrets: Dictionary containing regret lists for each strategy.
    """

    colors = ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600']

    plt.figure(figsize=(10, 6))
    for idx, (strategy, regret) in enumerate(cumulative_regrets.items()):
        plt.plot(regret, label=strategy, color=colors[idx % len(colors)], linewidth=2)

    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Plot total reward as a bar chart
    plt.figure(figsize=(10, 6))
    colors = ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600']

    plt.bar(strategy_names, rewards, color=colors[:len(strategy_names)])
    plt.xlabel('Strategy')
    plt.ylabel('Total Reward')
    plt.title('Comparison of Bandit Strategies')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_alpha_regret(cumulative_regrets):
    """
    Plots cumulative regret for different alpha values in Thompson Sampling.

    :param cumulative_regrets: Dictionary of regret lists for different alpha values.
    """
    colors = ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600']

    plt.figure(figsize=(10, 6))

    for idx, (alpha, regret) in enumerate(cumulative_regrets.items()):
        plt.plot(regret, label=f"α={alpha}", color=colors[idx % len(colors)], linewidth=2)

    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Effect of Alpha on Thompson Sampling Regret')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

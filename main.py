import numpy as np
from bandit.contextual_bandit import ContextualBandit
from utils import generate_synthetic_data, plot_results


def evaluate_bandit(strategy, batch_size=100, alpha_thompson=0.25):
    n_samples, n_features, n_actions, lambda_reg = 1000, 10, 5, 1.0
    epsilon = 0.1
    X, rewards = generate_synthetic_data(n_samples, n_features, n_actions, with_noise=False)

    bandit = ContextualBandit(n_features=10, n_actions=5, lambda_reg=0.1, alpha_thompson=alpha_thompson, alpha_linucb=0.5,
                              strategy=strategy)

    # bandit = ContextualBandit(n_features, n_actions, lambda_reg, strategy=strategy, alpha=alpha, epsilon=epsilon)
    cumulative_regret = []

    optimal_rewards = np.max(rewards, axis=1)
    accumulated_reward = 0

    # Lists to store batched data
    batch_contexts = []
    batch_actions = []
    batch_rewards = []

    for t in range(n_samples):
        context = X[t]
        best_action = np.argmax(rewards[t])
        action = bandit.select_action(context)
        reward = 1 if action == best_action else 0
        accumulated_reward += reward
        regret = np.sum(optimal_rewards[:t+1]) - accumulated_reward
        cumulative_regret.append(regret)

        # Store batch data
        batch_contexts.append(context)
        batch_actions.append(action)
        batch_rewards.append(reward)

        # Perform batch update
        if (t + 1) % batch_size == 0 or t == n_samples - 1:
            bandit.batch_update(np.array(batch_contexts), np.array(batch_actions), np.array(batch_rewards))
            batch_contexts, batch_actions, batch_rewards = [], [], []  # Clear batch

    strategies_with_alpha = ["thompson", "linucb", "epsilon-greedy"]

    print(f"Strategy: {strategy}, Total reward: {accumulated_reward}/{n_samples}")
    return strategy, accumulated_reward, cumulative_regret


if __name__ == "__main__":
    strategies = [
        "thompson",
        "linucb",
        "exploit",
        "epsilon-greedy",
        "random"
    ]

    results = []
    cumulative_regrets = {}

    for strategy in strategies:
        strategy_name, reward, regret = evaluate_bandit(strategy)
        results.append((strategy_name, reward))
        cumulative_regrets[strategy_name] = regret

    # Extract strategy names and rewards for plotting
    strategy_names, rewards = zip(*results)

    # Plot rewards and regret
    plot_results(strategy_names, rewards, cumulative_regrets)
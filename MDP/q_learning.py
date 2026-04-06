# q_learning.py

import numpy as np
from mdp_env import GridMDP

def q_learning(env, episodes=500, alpha=0.1, epsilon=0.3):
    """
    ── YOUR CUSTOMIZATION POINTS ─────────────────────────
    alpha   : learning rate       (try 0.01 – 0.5)
    epsilon : exploration rate    (try decay schedules)
    episodes: training episodes   (increase for harder envs)
    ──────────────────────────────────────────────────────
    """
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_log = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        # ── YOUR CUSTOMIZATION: epsilon decay ─────────────
        eps = max(0.05, epsilon * (0.99 ** ep))   # decay every episode

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < eps:
                action = np.random.randint(env.n_actions)   # explore
            else:
                action = np.argmax(Q[state])                # exploit

            next_state, reward, done = env.step(state, action)

            # ── CORE Q-UPDATE (Bellman) ─────────────────────
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (
                reward + env.gamma * best_next - Q[state, action]
            )
            # ────────────────────────────────────────────────

            state = next_state
            total_reward += reward

        rewards_log.append(total_reward)

    return Q, rewards_log


if __name__ == "__main__":
    env = GridMDP(grid_size=4, gamma=0.9)
    Q, rewards = q_learning(env, episodes=1000)

    policy = Q.argmax(axis=1)
    print("Learned policy:")
    print(policy.reshape(4, 4))
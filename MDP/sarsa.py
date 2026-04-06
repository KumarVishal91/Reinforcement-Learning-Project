# sarsa.py
import numpy as np
import matplotlib.pyplot as plt
from mdp_env import GridMDP

def sarsa(env, episodes=500, alpha=0.1, epsilon=0.3, gamma=0.9):
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_log = []
    steps_log = []

    def epsilon_greedy(state, eps):
        if np.random.rand() < eps:
            return np.random.randint(env.n_actions)
        return np.argmax(Q[state])

    for ep in range(episodes):
        state  = env.reset()
        eps    = max(0.05, epsilon * (0.99 ** ep))
        action = epsilon_greedy(state, eps)
        total_reward = 0
        done  = False
        steps = 0

        while not done:
            next_state, reward, done = env.step(state, action)

            # SARSA update (on-policy)
            next_action = epsilon_greedy(next_state, eps)
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            state        = next_state
            action       = next_action
            total_reward += reward
            steps        += 1

        rewards_log.append(total_reward)
        steps_log.append(steps)

    policy = Q.argmax(axis=1)
    return Q, rewards_log, steps_log, policy


if __name__ == "__main__":
    env = GridMDP()
    Q, rewards, steps, policy = sarsa(env, episodes=500)

    print("SARSA Learned policy:")
    print(policy.reshape(4, 4))

    # Plot steps per episode
    def smooth(data, window=20):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 4))
    plt.plot(smooth(steps), color='orange', linewidth=2)
    plt.title('Steps per Episode - SARSA (should decrease over time)')
    plt.xlabel('Episode')
    plt.ylabel('Steps taken')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sarsa_steps_per_episode.png')
    plt.show()
    print("Plot saved as sarsa_steps_per_episode.png")
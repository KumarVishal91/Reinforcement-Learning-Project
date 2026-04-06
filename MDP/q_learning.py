# q_learning.py
import numpy as np
import matplotlib.pyplot as plt
from mdp_env import GridMDP

def q_learning(env, episodes=500, alpha=0.1, epsilon=0.3):
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_log = []
    steps_log = []          # ← ADDED

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0           # ← ADDED

        # epsilon decay
        eps = max(0.05, epsilon * (0.99 ** ep))

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < eps:
                action = np.random.randint(env.n_actions)   # explore
            else:
                action = np.argmax(Q[state])                # exploit

            next_state, reward, done = env.step(state, action)

            # Q-Learning update (off-policy)
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (
                reward + env.gamma * best_next - Q[state, action]
            )

            state = next_state
            total_reward += reward
            steps += 1      # ← ADDED

        rewards_log.append(total_reward)
        steps_log.append(steps)     # ← ADDED

    policy = Q.argmax(axis=1)
    return Q, rewards_log, steps_log, policy    # ← ADDED steps_log


if __name__ == "__main__":
    env = GridMDP()
    Q, rewards, steps, policy = q_learning(env, episodes=500)

    print("Learned policy:")
    print(policy.reshape(4, 4))

    # ── Plot steps per episode ─────────────────────────────
    def smooth(data, window=20):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 4))
    plt.plot(smooth(steps), color='blue', linewidth=2)
    plt.title('Steps per Episode — Q-Learning (should decrease over time)')
    plt.xlabel('Episode')
    plt.ylabel('Steps taken')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('steps_per_episode.png')
    plt.show()
    print("Plot saved as steps_per_episode.png")
# compare.py
import numpy as np
import matplotlib.pyplot as plt
from mdp_env import GridMDP
from q_learning import q_learning
from sarsa import sarsa

def smooth(rewards, window=20):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

# Run both algorithms
env = GridMDP()
episodes = 500

_, q_rewards, q_policy   = q_learning(env, episodes=episodes)
_, s_rewards, s_policy   = sarsa(env,      episodes=episodes)

# ── Plot learning curves ──────────────────────────────────────
plt.figure(figsize=(10, 5))

plt.plot(smooth(q_rewards), label='Q-Learning', color='blue', linewidth=2)
plt.plot(smooth(s_rewards), label='SARSA',      color='orange', linewidth=2)

plt.xlabel('Episode')
plt.ylabel('Total Reward (smoothed)')
plt.title('SARSA vs Q-Learning — Learning Curve Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparison_plot.png')
plt.show()

# ── Print policy comparison ───────────────────────────────────
print("Q-Learning Policy:")
print(q_policy.reshape(4, 4))
print("\nSARSA Policy:")
print(s_policy.reshape(4, 4))
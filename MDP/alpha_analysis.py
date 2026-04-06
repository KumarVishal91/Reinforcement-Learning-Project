# alpha_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from mdp_env import GridMDP
from q_learning import q_learning

def smooth(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode='valid')

alphas = [0.01, 0.1, 0.3, 0.5, 0.9]
colors = ['blue', 'green', 'orange', 'red', 'purple']

env = GridMDP()
plt.figure(figsize=(10, 5))

for alpha, color in zip(alphas, colors):
    _, rewards, _, _ = q_learning(env, episodes=500, alpha=alpha)
    plt.plot(smooth(rewards), label=f'α={alpha}',
             color=color, linewidth=2)

plt.title('Effect of Learning Rate α on Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Total Reward (smoothed)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('alpha_analysis.png')
plt.show()
print("Saved alpha_analysis.png")
# visualize_policy.py
import numpy as np
import matplotlib.pyplot as plt
from mdp_env import GridMDP
from value_iteration import value_iteration

env = GridMDP()
V, policy = value_iteration(env)

# Arrow directions
arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Value function heatmap
im = axes[0].imshow(V.reshape(4,4), cmap='YlGn')
axes[0].set_title('Value Function Heatmap')
plt.colorbar(im, ax=axes[0])
for i in range(4):
    for j in range(4):
        s = i*4+j
        axes[0].text(j, i, f'{V[s]:.1f}',
                    ha='center', va='center', fontsize=10)

# Policy arrows
axes[1].imshow(np.zeros((4,4)), cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Optimal Policy')
for i in range(4):
    for j in range(4):
        s = i*4+j
        if s in env.obstacles:
            axes[1].text(j, i, 'X',
                        ha='center', va='center',
                        fontsize=16, color='red')
        elif s == env.goal_state:
            axes[1].text(j, i, 'G',
                        ha='center', va='center',
                        fontsize=16, color='green')
        else:
            axes[1].text(j, i, arrows[policy[s]],
                        ha='center', va='center', fontsize=16)

plt.tight_layout()
plt.savefig('policy_heatmap.png')
plt.show()
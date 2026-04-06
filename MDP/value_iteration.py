# value_iteration.py

import numpy as np
from mdp_env import GridMDP

def value_iteration(env, theta=1e-6, max_iter=1000):
    V = np.zeros(env.n_states)              # initialize V(s) = 0
    policy = np.zeros(env.n_states, dtype=int)

    for iteration in range(max_iter):
        delta = 0
        V_new = V.copy()

        for s in range(env.n_states):
            if s == env.goal_state:
                continue

            # ── YOUR UPDATE: Bellman optimality equation ──────────
            q_values = []
            for a in range(env.n_actions):
                s_next, r, _ = env.step(s, a)
                q = r + env.gamma * V[s_next]   # Q(s,a) = R + γ·V(s')
                q_values.append(q)

            V_new[s]   = max(q_values)
            policy[s]  = np.argmax(q_values)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < theta:                       # converged
            print(f"Converged at iteration {iteration}")
            break

    return V, policy


if __name__ == "__main__":
    env = GridMDP(grid_size=4, gamma=0.9)
    V, policy = value_iteration(env)

    print("Value function:")
    print(V.reshape(4, 4).round(2))

    print("\nOptimal policy (0=up 1=down 2=left 3=right):")
    print(policy.reshape(4, 4))
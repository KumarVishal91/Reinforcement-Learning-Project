# policy_iteration.py

import numpy as np
from mdp_env import GridMDP

def policy_evaluation(env, policy, V, theta=1e-6):
    """Evaluate a fixed policy until convergence."""
    while True:
        delta = 0
        for s in range(env.n_states):
            if s == env.goal_state:
                continue
            a = policy[s]
            s_next, r, _ = env.step(s, a)
            v_new  = r + env.gamma * V[s_next]   # ← YOUR UPDATE HERE
            delta  = max(delta, abs(v_new - V[s]))
            V[s]   = v_new
        if delta < theta:
            break
    return V

def policy_iteration(env):
    V      = np.zeros(env.n_states)
    policy = np.zeros(env.n_states, dtype=int)   # start with all action=0

    while True:
        V = policy_evaluation(env, policy, V)

        policy_stable = True
        for s in range(env.n_states):
            old_action = policy[s]

            # ── Greedy improvement step ───────────────────────────
            q_values  = [env.step(s, a)[1] + env.gamma * V[env.step(s, a)[0]]
                         for a in range(env.n_actions)]
            policy[s] = np.argmax(q_values)

            if old_action != policy[s]:
                policy_stable = False

        if policy_stable:
            break

    return V, policy
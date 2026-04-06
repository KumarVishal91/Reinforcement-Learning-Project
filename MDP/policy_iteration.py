import numpy as np
import matplotlib.pyplot as plt
from mdp_env import GridWorldMDP

# ─────────────────────────────────────────────
#  Policy Iteration (Upgraded)
#  Changes made:
#   1. Uses upgraded GridWorldMDP (obstacles + stochastic)
#   2. Tracks number of policy changes per iteration
#   3. Tracks V(s) during policy evaluation phase
#   4. Plots: policy changes per iter + value heatmap
#   5. Prints detailed per-iteration log
# ─────────────────────────────────────────────


def policy_evaluation(env, policy, theta=1e-6, max_iter=1000):
    """
    Evaluate a fixed policy until V(s) converges.

    V(s) = Σ_s' P(s'|s,π(s)) * [ R(s') + γ * V(s') ]

    Returns:
        V           : value function under this policy [n_states]
        eval_sweeps : number of sweeps needed to converge
    """
    V = np.zeros(env.n_states)

    for sweep in range(max_iter):
        delta = 0
        for s in range(env.n_states):
            if s in env.obstacle_states:
                continue

            v_old = V[s]
            a     = policy[s]          # fixed action from current policy

            # V(s) under fixed policy
            v_new = 0
            for s_next in range(env.n_states):
                prob = env.get_transition_prob(s, a, s_next)
                if prob > 0:
                    v_new += prob * (env.rewards[s_next] + env.gamma * V[s_next])

            V[s]  = v_new
            delta = max(delta, abs(v_old - v_new))

        if delta < theta:
            return V, sweep + 1

    return V, max_iter


# ─────────────────────────────────────────────────────────────────────────
def policy_improvement(env, V):
    """
    Greedily improve policy based on current V(s).

    π'(s) = argmax_a  Σ_s' P(s'|s,a) * [ R(s') + γ * V(s') ]

    Returns:
        new_policy    : improved policy [n_states]
        changed_count : number of states where action changed
    """
    new_policy     = np.zeros(env.n_states, dtype=int)
    changed_count  = 0

    for s in range(env.n_states):
        if s in env.obstacle_states or s == env.goal_state:
            continue

        action_values = []
        for a in range(env.n_actions):
            q_sa = 0
            for s_next in range(env.n_states):
                prob = env.get_transition_prob(s, a, s_next)
                if prob > 0:
                    q_sa += prob * (env.rewards[s_next] + env.gamma * V[s_next])
            action_values.append(q_sa)

        best_action    = np.argmax(action_values)
        new_policy[s]  = best_action

    return new_policy, changed_count


# ─────────────────────────────────────────────────────────────────────────
def policy_iteration(env, theta=1e-6):
    """
    Full Policy Iteration loop:
        Repeat until policy stops changing:
            1. Policy Evaluation  — compute V under current policy
            2. Policy Improvement — greedily update policy

    Returns:
        V              : final value function        [n_states]
        policy         : optimal policy              [n_states]
        changes_history: policy changes per iteration[n_iters]
        V_history      : V after each eval phase     [n_iters x n_states]
        sweeps_history : eval sweeps per iteration   [n_iters]
    """
    # ── Initialise with random policy ─────────────────────────────────────
    policy = np.random.randint(0, env.n_actions, size=env.n_states)

    changes_history = []   # NEW: policy changes per iteration
    V_history       = []   # NEW: V after each evaluation
    sweeps_history  = []   # NEW: how many sweeps evaluation needed

    print("=" * 65)
    print("        POLICY ITERATION — Iteration Log")
    print("=" * 65)
    print(f"{'Iter':>5}  {'Eval Sweeps':>12}  {'States Changed':>15}  {'Stable?':>8}")
    print("-" * 65)

    for iteration in range(100):   # max 100 policy iterations

        # ── Step 1: Policy Evaluation ─────────────────────────────────────
        V, eval_sweeps = policy_evaluation(env, policy, theta=theta)

        # ── Step 2: Policy Improvement ────────────────────────────────────
        new_policy, _ = policy_improvement(env, V)

        # ── Count how many states changed ─────────────────────────────────
        # NEW: track changed states explicitly
        changed = np.sum(new_policy != policy)
        for s in range(env.n_states):
            if s in env.obstacle_states or s == env.goal_state:
                continue

        changed = int(np.sum([
            new_policy[s] != policy[s]
            for s in range(env.n_states)
            if s not in env.obstacle_states and s != env.goal_state
        ]))

        # ── Record history ─────────────────────────────────────────────────
        changes_history.append(changed)
        V_history.append(V.copy())
        sweeps_history.append(eval_sweeps)

        stable = changed == 0
        print(f"{iteration+1:>5}  {eval_sweeps:>12}  {changed:>15}  {'YES ✓' if stable else 'no':>8}")

        # ── Update policy ──────────────────────────────────────────────────
        policy = new_policy

        # ── Stop if policy is stable ───────────────────────────────────────
        if stable:
            print("-" * 65)
            print(f"  Policy stable at iteration {iteration + 1}  —  Optimal policy found!")
            break

    print("=" * 65)
    return V, policy, changes_history, np.array(V_history), sweeps_history


# ─────────────────────────────────────────────────────────────────────────
def plot_policy_changes(changes_history, sweeps_history):
    """
    NEW: Two-panel plot showing:
      Left  — policy changes per iteration (should reach 0)
      Right — evaluation sweeps per iteration
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: policy changes
    axes[0].bar(range(1, len(changes_history) + 1), changes_history,
                color='steelblue', edgecolor='navy', alpha=0.8)
    axes[0].set_xlabel("Policy Iteration", fontsize=12)
    axes[0].set_ylabel("States Whose Action Changed", fontsize=12)
    axes[0].set_title("Policy Changes per Iteration\n(Reaches 0 = Optimal Policy Found)",
                       fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(1, len(changes_history) + 1))
    axes[0].grid(True, axis='y', alpha=0.4)

    # Right: evaluation sweeps
    axes[1].plot(range(1, len(sweeps_history) + 1), sweeps_history,
                 color='tomato', linewidth=2, marker='s', markersize=7)
    axes[1].set_xlabel("Policy Iteration", fontsize=12)
    axes[1].set_ylabel("Evaluation Sweeps Needed", fontsize=12)
    axes[1].set_title("Policy Evaluation Cost per Iteration\n(Sweeps until V(s) converged)",
                       fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(1, len(sweeps_history) + 1))
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("policy_changes.png", dpi=150)
    plt.show()
    print("Saved: policy_changes.png")


# ─────────────────────────────────────────────────────────────────────────
def plot_value_heatmap(env, V, policy, title="Policy Iteration — Final V(s) + Optimal Policy"):
    """
    Heatmap of final V(s) + policy arrows.
    """
    grid = V.reshape(env.grid_size, env.grid_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap='PuBuGn', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='V(s) — State Value')

    arrow_map = {0: (0, -0.35), 1: (0, 0.35), 2: (-0.35, 0), 3: (0.35, 0)}

    for s in range(env.n_states):
        row, col = env._state_to_rc(s)

        if s in env.obstacle_states:
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color='#2c2c2c'))
            ax.text(col, row, 'WALL', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        elif s == env.goal_state:
            ax.text(col, row, f'GOAL\n{V[s]:.1f}', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        else:
            ax.text(col, row + 0.25, f'{V[s]:.2f}', ha='center', va='center',
                    fontsize=8, color='#333')
            dx, dy = arrow_map[policy[s]]
            ax.annotate("", xy=(col + dx, row + dy),
                        xytext=(col, row),
                        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.8))

    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.set_xticklabels([f'Col {i}' for i in range(env.grid_size)])
    ax.set_yticklabels([f'Row {i}' for i in range(env.grid_size)])
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("pi_value_heatmap.png", dpi=150)
    plt.show()
    print("Saved: pi_value_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────
def plot_v_evolution(V_history, env):
    """
    NEW: Show how V(s) evolves across policy iterations
    for a few key states.
    """
    states_to_track = [0, 3, 6, 9, 12, 14]
    n_iters = len(V_history)
    colors  = plt.cm.tab10(np.linspace(0, 1, len(states_to_track)))

    plt.figure(figsize=(9, 5))
    for i, s in enumerate(states_to_track):
        values = [V_history[it][s] for it in range(n_iters)]
        plt.plot(range(1, n_iters + 1), values,
                 label=f'State {s}', color=colors[i],
                 linewidth=2, marker='o', markersize=6)

    plt.xlabel("Policy Iteration", fontsize=12)
    plt.ylabel("V(s)", fontsize=12)
    plt.title("Policy Iteration — V(s) Evolution Across Iterations", fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.xticks(range(1, n_iters + 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pi_value_evolution.png", dpi=150)
    plt.show()
    print("Saved: pi_value_evolution.png")


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    env = GridWorldMDP(grid_size=4, slip_prob=0.1, gamma=0.9)

    # ── Run Policy Iteration ──────────────────────────────────────────────
    V, policy, changes_history, V_history, sweeps_history = policy_iteration(env)

    # ── Print results ─────────────────────────────────────────────────────
    print("\nFinal Value Function V(s):")
    print(V.reshape(env.grid_size, env.grid_size).round(3))

    print("\nOptimal Policy π(s)  [0=UP 1=DOWN 2=LEFT 3=RIGHT]:")
    print(policy.reshape(env.grid_size, env.grid_size))

    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
    print("\nOptimal Policy (readable):")
    for s in range(env.n_states):
        row, col = env._state_to_rc(s)
        if s in env.obstacle_states:
            label = "WALL"
        elif s == env.goal_state:
            label = "GOAL"
        else:
            label = action_names[policy[s]]
        print(f"  State {s:2d} (row={row}, col={col}) → {label}")

    print(f"\nTotal policy iterations : {len(changes_history)}")
    print(f"Total evaluation sweeps : {sum(sweeps_history)}")
    print(f"Policy changes history  : {changes_history}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_policy_changes(changes_history, sweeps_history)
    plot_value_heatmap(env, V, policy)
    plot_v_evolution(V_history, env)
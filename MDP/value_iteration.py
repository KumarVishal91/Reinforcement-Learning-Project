import numpy as np
import matplotlib.pyplot as plt
from mdp_env import GridWorldMDP

# ─────────────────────────────────────────────
#  Value Iteration (Upgraded)
#  Changes made:
#   1. Uses upgraded GridWorldMDP (obstacles + stochastic)
#   2. Tracks delta (max change) at every sweep → convergence history
#   3. Tracks V(s) history for every state over sweeps
#   4. Plots: convergence curve + value heatmap + policy arrows
#   5. Prints detailed iteration-by-iteration log
# ─────────────────────────────────────────────

def value_iteration(env, theta=1e-6, max_iterations=1000):
    """
    Perform Value Iteration on the MDP environment.

    Parameters:
        env            : GridWorldMDP instance
        theta          : convergence threshold (stop when delta < theta)
        max_iterations : safety cap on iterations

    Returns:
        V              : final value function array  [n_states]
        policy         : optimal policy array        [n_states]
        delta_history  : max delta per sweep         [n_iterations]
        V_history      : V values per sweep          [n_iterations x n_states]
    """

    # ── Initialise V(s) = 0 for all states ───────────────────────────────
    V = np.zeros(env.n_states)

    # NEW: track delta and V at each sweep
    delta_history = []
    V_history     = []

    print("=" * 55)
    print("       VALUE ITERATION — Convergence Log")
    print("=" * 55)
    print(f"{'Iter':>5}  {'Max Delta':>12}  {'Converged?':>12}")
    print("-" * 55)

    for iteration in range(max_iterations):
        delta = 0   # max change across all states this sweep

        for s in range(env.n_states):

            # Skip obstacle states — no value to compute
            if s in env.obstacle_states:
                continue

            v_old = V[s]

            # ── Bellman optimality update ─────────────────────────────────
            # V(s) = max_a  Σ_s'  P(s'|s,a) * [ R(s') + γ * V(s') ]
            action_values = []
            for a in range(env.n_actions):
                q_sa = 0
                for s_next in range(env.n_states):
                    prob = env.get_transition_prob(s, a, s_next)
                    if prob > 0:
                        q_sa += prob * (env.rewards[s_next] + env.gamma * V[s_next])
                action_values.append(q_sa)

            V[s] = max(action_values)

            # Track max change
            delta = max(delta, abs(v_old - V[s]))

        # ── Record history ────────────────────────────────────────────────
        delta_history.append(delta)
        V_history.append(V.copy())

        converged = delta < theta
        print(f"{iteration+1:>5}  {delta:>12.6f}  {'YES ✓' if converged else 'no':>12}")

        if converged:
            print("-" * 55)
            print(f"  Converged at iteration {iteration + 1}  (delta < {theta})")
            break

    print("=" * 55)

    # ── Extract optimal policy ────────────────────────────────────────────
    policy = extract_policy(env, V)

    return V, policy, delta_history, np.array(V_history)


# ─────────────────────────────────────────────────────────────────────────
def extract_policy(env, V):
    """
    Extract greedy policy from value function V.
    policy[s] = argmax_a  Σ_s' P(s'|s,a)[R(s') + γV(s')]
    """
    policy = np.zeros(env.n_states, dtype=int)

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

        policy[s] = np.argmax(action_values)

    return policy


# ─────────────────────────────────────────────────────────────────────────
def plot_convergence(delta_history):
    """
    NEW: Plot how delta (max change) decreases each sweep.
    This directly shows convergence speed of Value Iteration.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(delta_history) + 1), delta_history,
             color='steelblue', linewidth=2, marker='o', markersize=4)
    plt.yscale('log')   # log scale shows exponential decay clearly
    plt.xlabel("Iteration (Sweep)", fontsize=12)
    plt.ylabel("Max Delta  (log scale)", fontsize=12)
    plt.title("Value Iteration — Convergence Curve\n(Max change in V(s) per sweep)", fontsize=13)
    plt.grid(True, alpha=0.4, which='both')
    plt.tight_layout()
    plt.savefig("vi_convergence.png", dpi=150)
    plt.show()
    print("Saved: vi_convergence.png")


# ─────────────────────────────────────────────────────────────────────────
def plot_value_heatmap(env, V, policy):
    """
    NEW: Heatmap of final V(s) values + policy arrows overlaid.
    """
    grid = V.reshape(env.grid_size, env.grid_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap='YlGnBu', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='V(s) — State Value')

    arrow_map = {
        0: (0, -0.35),    # UP    → arrow points up   (row decreases)
        1: (0,  0.35),    # DOWN
        2: (-0.35, 0),    # LEFT
        3: ( 0.35, 0),    # RIGHT
    }

    for s in range(env.n_states):
        row, col = env._state_to_rc(s)

        if s in env.obstacle_states:
            ax.text(col, row, 'WALL', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
            ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1,
                                        color='#2c2c2c'))
        elif s == env.goal_state:
            ax.text(col, row, f'GOAL\n{V[s]:.1f}', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        else:
            # Value text
            ax.text(col, row + 0.25, f'{V[s]:.2f}', ha='center', va='center',
                    fontsize=8, color='#333')
            # Policy arrow
            dx, dy = arrow_map[policy[s]]
            ax.annotate("", xy=(col + dx, row + dy),
                        xytext=(col, row),
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.8))

    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.set_xticklabels([f'Col {i}' for i in range(env.grid_size)])
    ax.set_yticklabels([f'Row {i}' for i in range(env.grid_size)])
    ax.set_title("Value Iteration — V(s) Heatmap + Optimal Policy", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("vi_value_heatmap.png", dpi=150)
    plt.show()
    print("Saved: vi_value_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────
def plot_value_evolution(V_history, states_to_track=None):
    """
    NEW: Show how V(s) for selected states evolves over iterations.
    Helps understand which states take longer to converge.
    """
    n_iters = len(V_history)
    if states_to_track is None:
        states_to_track = [0, 3, 6, 9, 12, 15]   # sample states

    plt.figure(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(states_to_track)))

    for i, s in enumerate(states_to_track):
        values = [V_history[it][s] for it in range(n_iters)]
        plt.plot(range(1, n_iters + 1), values,
                 label=f'State {s}', color=colors[i], linewidth=2)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("V(s)", fontsize=12)
    plt.title("Value Iteration — V(s) Evolution per State\n(How each state's value stabilises)", fontsize=13)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("vi_value_evolution.png", dpi=150)
    plt.show()
    print("Saved: vi_value_evolution.png")


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Create environment ────────────────────────────────────────────────
    env = GridWorldMDP(grid_size=4, slip_prob=0.1, gamma=0.9)

    # ── Run Value Iteration ───────────────────────────────────────────────
    V, policy, delta_history, V_history = value_iteration(env, theta=1e-6)

    # ── Print results ─────────────────────────────────────────────────────
    print("\nFinal Value Function V(s):")
    print(V.reshape(env.grid_size, env.grid_size).round(3))

    print("\nOptimal Policy π(s)  [0=UP 1=DOWN 2=LEFT 3=RIGHT]:")
    print(policy.reshape(env.grid_size, env.grid_size))

    action_names = {0:'UP', 1:'DOWN', 2:'LEFT', 3:'RIGHT'}
    print("\nOptimal Policy (readable):")
    for s in range(env.n_states):
        row, col = env._state_to_rc(s)
        if s in env.obstacle_states:
            label = "WALL"
        elif s == env.goal_state:
            label = "GOAL"
        else:
            label = action_names[policy[s]]
        print(f"  State {s:2d} (row={row},col={col}) → {label}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_convergence(delta_history)
    plot_value_heatmap(env, V, policy)
    plot_value_evolution(V_history)

    print(f"\nTotal iterations to converge : {len(delta_history)}")
    print(f"Final delta                  : {delta_history[-1]:.2e}")
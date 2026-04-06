import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from mdp_env import GridWorldMDP

# ─────────────────────────────────────────────
#  visualize_policy.py  (Upgraded)
#  Changes made:
#   1. Shows Value Function heatmap alongside Policy
#   2. Draws clean directional arrows for each state
#   3. Highlights START, GOAL, WALL cells with colors
#   4. Adds convergence iteration tracker display
#   5. Side-by-side: V(s) heatmap  |  Policy grid
#   6. Saves both combined and individual plots
# ─────────────────────────────────────────────


# ── Inline value iteration (so this file is self-contained) ──────────────
def run_value_iteration(env, theta=1e-6, max_iterations=1000):
    V = np.zeros(env.n_states)
    converged_at = max_iterations

    for iteration in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            if s in env.obstacle_states:
                continue
            v_old = V[s]
            action_values = []
            for a in range(env.n_actions):
                q_sa = 0
                for s_next in range(env.n_states):
                    prob = env.get_transition_prob(s, a, s_next)
                    if prob > 0:
                        q_sa += prob * (env.rewards[s_next] + env.gamma * V[s_next])
                action_values.append(q_sa)
            V[s]  = max(action_values)
            delta = max(delta, abs(v_old - V[s]))

        if delta < theta:
            converged_at = iteration + 1
            print(f"  Value Iteration converged at iteration {converged_at}")
            break

    return V, converged_at


def extract_policy(env, V):
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
def plot_combined(env, V, policy, converged_at):
    """
    NEW: Main combined visualization.
    Left panel  — V(s) heatmap with values annotated
    Right panel — Policy grid with directional arrows
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"MDP Grid World  |  γ={env.gamma}  |  Slip={env.slip_prob}  "
        f"|  Converged at iteration {converged_at}",
        fontsize=13, fontweight='bold', y=1.01
    )

    # ── Arrow direction map: action → (dx, dy) in plot coords ────────────
    # imshow has y-axis flipped so UP = negative dy
    arrow_map = {
        0: ( 0.00, -0.30),   # UP
        1: ( 0.00,  0.30),   # DOWN
        2: (-0.30,  0.00),   # LEFT
        3: ( 0.30,  0.00),   # RIGHT
    }
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  LEFT PANEL — Value Function Heatmap
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax = axes[0]
    grid_V = V.reshape(env.grid_size, env.grid_size).copy()

    # Mask obstacle cells so they don't pollute colormap
    masked = np.ma.masked_where(
        np.isin(np.arange(env.n_states).reshape(env.grid_size, env.grid_size),
                list(env.obstacle_states)),
        grid_V
    )

    cmap = plt.cm.YlGnBu
    cmap.set_bad(color='#1a1a2e')   # obstacle color

    im = ax.imshow(masked, cmap=cmap, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('V(s) — State Value', fontsize=10)

    for s in range(env.n_states):
        row = s // env.grid_size
        col = s % env.grid_size

        if s in env.obstacle_states:
            ax.text(col, row, '■ WALL', ha='center', va='center',
                    fontsize=10, color='#aaaaaa', fontweight='bold')

        elif s == env.goal_state:
            ax.add_patch(mpatches.FancyBboxPatch(
                (col - 0.45, row - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05", color='#27ae60', zorder=2
            ))
            ax.text(col, row, f'★ GOAL\n{V[s]:.2f}',
                    ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold', zorder=3)

        elif s == env.start_state:
            ax.add_patch(mpatches.FancyBboxPatch(
                (col - 0.45, row - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05", color='#2980b9', zorder=2, alpha=0.5
            ))
            ax.text(col, row, f'▶ START\n{V[s]:.2f}',
                    ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold', zorder=3)

        else:
            ax.text(col, row, f'{V[s]:.2f}',
                    ha='center', va='center', fontsize=11,
                    color='#1a1a2e', fontweight='bold')

    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.set_xticklabels([f'Col {i}' for i in range(env.grid_size)], fontsize=9)
    ax.set_yticklabels([f'Row {i}' for i in range(env.grid_size)], fontsize=9)
    ax.set_title('Value Function  V(s)', fontsize=12, fontweight='bold', pad=10)

    # Grid lines
    for x in np.arange(-0.5, env.grid_size, 1):
        ax.axhline(x, color='white', lw=1.5)
        ax.axvline(x, color='white', lw=1.5)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  RIGHT PANEL — Policy Grid with Arrows
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax2 = axes[1]

    # Light pastel background grid
    bg_grid = np.zeros((env.grid_size, env.grid_size))
    ax2.imshow(bg_grid, cmap='Greys', vmin=0, vmax=1,
               interpolation='nearest', alpha=0.05)

    for s in range(env.n_states):
        row = s // env.grid_size
        col = s % env.grid_size

        if s in env.obstacle_states:
            ax2.add_patch(plt.Rectangle(
                (col - 0.5, row - 0.5), 1, 1,
                color='#2c2c2c', zorder=2
            ))
            ax2.text(col, row, '■\nWALL', ha='center', va='center',
                     fontsize=9, color='#888888', fontweight='bold', zorder=3)

        elif s == env.goal_state:
            ax2.add_patch(plt.Rectangle(
                (col - 0.5, row - 0.5), 1, 1,
                color='#27ae60', zorder=2
            ))
            ax2.text(col, row, '★\nGOAL', ha='center', va='center',
                     fontsize=11, color='white', fontweight='bold', zorder=3)

        elif s == env.start_state:
            ax2.add_patch(plt.Rectangle(
                (col - 0.5, row - 0.5), 1, 1,
                color='#2980b9', alpha=0.3, zorder=2
            ))
            dx, dy = arrow_map[policy[s]]
            ax2.annotate(
                "", xy=(col + dx, row + dy), xytext=(col, row),
                arrowprops=dict(arrowstyle="-|>", color='#2980b9',
                                lw=2.5, mutation_scale=20),
                zorder=4
            )
            ax2.text(col, row - 0.38, '▶ START', ha='center', va='center',
                     fontsize=7, color='#2980b9', fontstyle='italic')
            ax2.text(col + dx * 0.1, row + dy * 0.1 + 0.25,
                     action_symbols[policy[s]],
                     ha='center', va='center', fontsize=14,
                     color='#2980b9', fontweight='bold', zorder=5)

        else:
            # Normal state: draw arrow
            dx, dy = arrow_map[policy[s]]
            ax2.annotate(
                "", xy=(col + dx, row + dy), xytext=(col, row),
                arrowprops=dict(arrowstyle="-|>", color='#c0392b',
                                lw=2.5, mutation_scale=20),
                zorder=4
            )
            # Symbol label
            ax2.text(col, row + 0.3, action_symbols[policy[s]],
                     ha='center', va='center', fontsize=16,
                     color='#c0392b', fontweight='bold', zorder=5)
            # State number
            ax2.text(col, row - 0.33, f'S{s}',
                     ha='center', va='center', fontsize=7.5,
                     color='#555', zorder=5)

    # Grid lines
    for x in np.arange(-0.5, env.grid_size, 1):
        ax2.axhline(x, color='#cccccc', lw=1.2)
        ax2.axvline(x, color='#cccccc', lw=1.2)

    ax2.set_xlim(-0.5, env.grid_size - 0.5)
    ax2.set_ylim(-0.5, env.grid_size - 0.5)
    ax2.set_xticks(range(env.grid_size))
    ax2.set_yticks(range(env.grid_size))
    ax2.set_xticklabels([f'Col {i}' for i in range(env.grid_size)], fontsize=9)
    ax2.set_yticklabels([f'Row {i}' for i in range(env.grid_size)], fontsize=9)
    ax2.set_title('Optimal Policy  π(s)', fontsize=12, fontweight='bold', pad=10)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color='#2980b9', alpha=0.5, label='Start State'),
        mpatches.Patch(color='#27ae60',             label='Goal State'),
        mpatches.Patch(color='#2c2c2c',             label='Obstacle (Wall)'),
        mpatches.Patch(color='#c0392b', alpha=0.7,  label='Policy Arrow'),
    ]
    ax2.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9,
               framealpha=0.9)

    plt.tight_layout()
    plt.savefig("value_and_policy.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: value_and_policy.png")


# ─────────────────────────────────────────────────────────────────────────
def plot_policy_only(env, policy):
    """
    Clean standalone policy grid — one action arrow per state.
    """
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    action_colors  = {0: '#3498db', 1: '#e74c3c', 2: '#2ecc71', 3: '#f39c12'}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_aspect('equal')

    for s in range(env.n_states):
        row = s // env.grid_size
        col = s % env.grid_size
        a   = policy[s]

        if s in env.obstacle_states:
            ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, color='#2c2c2c'))
            ax.text(col, row, 'WALL', ha='center', va='center',
                    fontsize=10, color='white', fontweight='bold')
        elif s == env.goal_state:
            ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, color='#27ae60'))
            ax.text(col, row, '★\nGOAL', ha='center', va='center',
                    fontsize=11, color='white', fontweight='bold')
        else:
            ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1,
                                        color=action_colors[a], alpha=0.15))
            ax.text(col, row, action_symbols[a], ha='center', va='center',
                    fontsize=28, color=action_colors[a], fontweight='bold')
            ax.text(col + 0.38, row - 0.38, f'S{s}', ha='right', va='bottom',
                    fontsize=7, color='#777')

    for x in np.arange(-0.5, env.grid_size, 1):
        ax.axhline(x, color='#aaa', lw=1)
        ax.axvline(x, color='#aaa', lw=1)

    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.set_xticklabels([f'Col {i}' for i in range(env.grid_size)])
    ax.set_yticklabels([f'Row {i}' for i in range(env.grid_size)])
    ax.set_title('Optimal Policy  π(s)\n[↑UP  ↓DOWN  ←LEFT  →RIGHT]',
                 fontsize=12, fontweight='bold')

    # Color legend
    legend_elements = [
        mpatches.Patch(color='#3498db', alpha=0.6, label='↑ UP'),
        mpatches.Patch(color='#e74c3c', alpha=0.6, label='↓ DOWN'),
        mpatches.Patch(color='#2ecc71', alpha=0.6, label='← LEFT'),
        mpatches.Patch(color='#f39c12', alpha=0.6, label='→ RIGHT'),
    ]
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9)

    plt.tight_layout()
    plt.savefig("policy_grid.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: policy_grid.png")


# ─────────────────────────────────────────────────────────────────────────
def plot_value_only(env, V):
    """
    Standalone V(s) heatmap with annotated values.
    """
    grid = V.reshape(env.grid_size, env.grid_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='V(s)')

    for s in range(env.n_states):
        row = s // env.grid_size
        col = s % env.grid_size
        if s in env.obstacle_states:
            ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, color='#1a1a2e'))
            ax.text(col, row, 'WALL', ha='center', va='center',
                    fontsize=9, color='#aaa', fontweight='bold')
        elif s == env.goal_state:
            ax.text(col, row, f'GOAL\n{V[s]:.2f}',
                    ha='center', va='center', fontsize=9,
                    color='white', fontweight='bold')
        else:
            ax.text(col, row, f'{V[s]:.3f}',
                    ha='center', va='center', fontsize=10, color='#111')

    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.set_xticklabels([f'Col {i}' for i in range(env.grid_size)])
    ax.set_yticklabels([f'Row {i}' for i in range(env.grid_size)])
    ax.set_title('State Value Function  V(s)',
                 fontsize=12, fontweight='bold')

    for x in np.arange(-0.5, env.grid_size, 1):
        ax.axhline(x, color='white', lw=1)
        ax.axvline(x, color='white', lw=1)

    plt.tight_layout()
    plt.savefig("value_heatmap_only.png", dpi=150)
    plt.show()
    print("Saved: value_heatmap_only.png")


# ─────────────────────────────────────────────────────────────────────────
def print_policy_table(env, V, policy):
    """
    Print a readable summary table of the final policy and values.
    """
    action_names = {0: 'UP   ', 1: 'DOWN ', 2: 'LEFT ', 3: 'RIGHT'}

    print("\n" + "=" * 58)
    print(f"  {'State':>6}  {'(Row,Col)':>10}  {'V(s)':>8}  {'Action':>8}  {'Type':>8}")
    print("-" * 58)

    for s in range(env.n_states):
        row, col = divmod(s, env.grid_size)
        if s in env.obstacle_states:
            stype  = 'WALL'
            action = '  —  '
        elif s == env.goal_state:
            stype  = 'GOAL'
            action = '  —  '
        elif s == env.start_state:
            stype  = 'START'
            action = action_names[policy[s]]
        else:
            stype  = 'normal'
            action = action_names[policy[s]]

        print(f"  {s:>6}  ({row},{col}){' ':>7}  {V[s]:>8.3f}  {action:>8}  {stype:>8}")

    print("=" * 58)


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("   VISUALIZE POLICY  (Upgraded)")
    print("=" * 50)

    # ── Environment ───────────────────────────────────────────────────────
    env = GridWorldMDP(grid_size=4, slip_prob=0.1, gamma=0.9)

    print(f"\nEnvironment:")
    print(f"  Grid      : {env.grid_size}x{env.grid_size}  ({env.n_states} states)")
    print(f"  Obstacles : {env.obstacle_states}")
    print(f"  Slip prob : {env.slip_prob}")
    print(f"  Gamma     : {env.gamma}")
    print()

    # ── Run Value Iteration ───────────────────────────────────────────────
    V, converged_at = run_value_iteration(env)
    policy = extract_policy(env, V)

    # ── Print table ───────────────────────────────────────────────────────
    print_policy_table(env, V, policy)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating visualizations...")
    plot_combined(env, V, policy, converged_at)   # main combined plot
    plot_policy_only(env, policy)                  # standalone policy
    plot_value_only(env, V)                        # standalone heatmap

    print("\nAll files saved:")
    print("  value_and_policy.png     ← main combined plot (NEW)")
    print("  policy_grid.png          ← color-coded policy arrows (NEW)")
    print("  value_heatmap_only.png   ← standalone V(s) heatmap  (NEW)")
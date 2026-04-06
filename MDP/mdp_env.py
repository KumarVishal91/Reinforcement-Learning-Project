import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─────────────────────────────────────────────
#  Grid World MDP Environment (Upgraded)
#  Changes made:
#   1. Added obstacle states (impassable cells)
#   2. Added stochastic (slippery) transitions
#   3. Added step counter and episode tracking
#   4. Added render() method for visualization
# ─────────────────────────────────────────────

class GridWorldMDP:
    def __init__(self, grid_size=4, slip_prob=0.1, gamma=0.9):
        """
        Parameters:
            grid_size : int   - NxN grid (default 4x4 = 16 states)
            slip_prob : float - probability of random action (stochastic)
            gamma     : float - discount factor
        """
        self.grid_size  = grid_size
        self.n_states   = grid_size * grid_size
        self.n_actions  = 4          # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.slip_prob  = slip_prob  # NEW: stochastic transition probability
        self.gamma      = gamma

        # ── Actions ──────────────────────────────────────────────────────
        # Each action shifts row/col by these deltas
        self.action_deltas = {
            0: (-1,  0),   # UP
            1: ( 1,  0),   # DOWN
            2: ( 0, -1),   # LEFT
            3: ( 0,  1),   # RIGHT
        }
        self.action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

        # ── Special States ────────────────────────────────────────────────
        self.start_state = 0                          # top-left
        self.goal_state  = self.n_states - 1          # bottom-right (state 15)

        # NEW: Obstacle states — agent cannot enter these cells
        # States 5 and 10 in a 4x4 grid (interior cells)
        self.obstacle_states = {5, 10}

        # ── Rewards ───────────────────────────────────────────────────────
        self.reward_goal     =  10   # reached goal
        self.reward_obstacle = -5    # tried to enter obstacle (bounced back)
        self.reward_step     = -1    # every other step

        # ── Build transition & reward tables ─────────────────────────────
        self.transitions = self._build_transitions()
        self.rewards     = self._build_rewards()

        # ── Episode state ─────────────────────────────────────────────────
        self.current_state = self.start_state
        self.steps         = 0
        self.max_steps     = 100

    # ─────────────────────────────────────────────────────────────────────
    def _state_to_rc(self, state):
        """Convert flat state index → (row, col)"""
        return divmod(state, self.grid_size)

    def _rc_to_state(self, row, col):
        """Convert (row, col) → flat state index"""
        return row * self.grid_size + col

    # ─────────────────────────────────────────────────────────────────────
    def _build_transitions(self):
        """
        Build deterministic next-state table.
        Stochastic slip is applied at runtime in step().
        transitions[s][a] = next_state (without slip)
        """
        T = {}
        for s in range(self.n_states):
            T[s] = {}
            row, col = self._state_to_rc(s)
            for a, (dr, dc) in self.action_deltas.items():
                new_row = row + dr
                new_col = col + dc
                # Stay in bounds
                if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                    next_s = self._rc_to_state(new_row, new_col)
                else:
                    next_s = s   # hit wall → stay
                T[s][a] = next_s
        return T

    # ─────────────────────────────────────────────────────────────────────
    def _build_rewards(self):
        """
        Build reward table: rewards[s] = reward for being in state s.
        """
        R = {}
        for s in range(self.n_states):
            if s == self.goal_state:
                R[s] = self.reward_goal
            elif s in self.obstacle_states:
                R[s] = self.reward_obstacle
            else:
                R[s] = self.reward_step
        return R

    # ─────────────────────────────────────────────────────────────────────
    def reset(self):
        """Reset episode — returns starting state."""
        self.current_state = self.start_state
        self.steps         = 0
        return self.current_state

    # ─────────────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Take one step in the environment.

        NEW — Stochastic transitions:
            With probability slip_prob, a random action is taken instead
            of the intended one (simulates wind/uncertainty in real MDPs).

        Returns: (next_state, reward, done)
        """
        assert action in range(self.n_actions), f"Invalid action {action}"

        # ── Stochastic slip ───────────────────────────────────────────────
        if np.random.rand() < self.slip_prob:
            action = np.random.choice(self.n_actions)   # random action (slip)

        # ── Deterministic transition ──────────────────────────────────────
        next_state = self.transitions[self.current_state][action]

        # ── Obstacle check: bounce back ───────────────────────────────────
        if next_state in self.obstacle_states:
            next_state = self.current_state  # stay where you are

        # ── Reward ────────────────────────────────────────────────────────
        reward = self.rewards[next_state]

        # ── Done condition ────────────────────────────────────────────────
        self.steps += 1
        done = (next_state == self.goal_state) or (self.steps >= self.max_steps)

        self.current_state = next_state
        return next_state, reward, done

    # ─────────────────────────────────────────────────────────────────────
    def get_transition_prob(self, s, a, s_next):
        """
        Return P(s_next | s, a) — used by model-based algorithms.

        With slip_prob, any of the 4 actions could be taken.
        """
        prob = 0.0
        for a_actual in range(self.n_actions):
            # Probability of taking a_actual instead of a
            if a_actual == a:
                p_action = 1.0 - self.slip_prob + self.slip_prob / self.n_actions
            else:
                p_action = self.slip_prob / self.n_actions

            intended_next = self.transitions[s][a_actual]

            # Obstacle bounce-back
            if intended_next in self.obstacle_states:
                intended_next = s

            if intended_next == s_next:
                prob += p_action

        return prob

    # ─────────────────────────────────────────────────────────────────────
    def render(self, V=None, policy=None, title="Grid World MDP"):
        """
        Visualize the grid with optional value function and policy arrows.
        NEW method added in this upgrade.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')

        arrow_map = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}

        for s in range(self.n_states):
            row, col = self._state_to_rc(s)
            # Flip row so row=0 is at top
            draw_row = self.grid_size - 1 - row

            # ── Cell color ────────────────────────────────────────────────
            if s in self.obstacle_states:
                color = '#2c2c2c'    # dark = obstacle
            elif s == self.goal_state:
                color = '#2ecc71'    # green = goal
            elif s == self.start_state:
                color = '#3498db'    # blue = start
            elif V is not None:
                # Heat map based on value
                norm_v = (V[s] - V.min()) / (V.max() - V.min() + 1e-9)
                color = plt.cm.YlOrRd(norm_v)
            else:
                color = '#ecf0f1'

            rect = patches.Rectangle(
                (col, draw_row), 1, 1,
                linewidth=1.5, edgecolor='#333', facecolor=color
            )
            ax.add_patch(rect)

            # ── Labels ────────────────────────────────────────────────────
            label = f"S{s}"
            if s == self.goal_state:
                label = "GOAL"
            elif s == self.start_state:
                label = "START"
            elif s in self.obstacle_states:
                label = "WALL"

            val_text = f"\nV={V[s]:.1f}" if V is not None and s not in self.obstacle_states else ""
            ax.text(col + 0.5, draw_row + 0.65, label + val_text,
                    ha='center', va='center', fontsize=7.5,
                    color='white' if s in self.obstacle_states or s == self.goal_state else '#333')

            # ── Policy arrows ─────────────────────────────────────────────
            if policy is not None and s not in self.obstacle_states and s != self.goal_state:
                dx, dy = arrow_map[policy[s]]
                ax.annotate("", xy=(col + 0.5 + dx, draw_row + 0.5 + dy),
                            xytext=(col + 0.5, draw_row + 0.5),
                            arrowprops=dict(arrowstyle="->", color="navy", lw=1.5))

        plt.tight_layout()
        plt.savefig("grid_render.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: grid_render.png")


# ─────────────────────────────────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = GridWorldMDP(grid_size=4, slip_prob=0.1, gamma=0.9)

    print("=== GridWorldMDP (Upgraded) ===")
    print(f"States      : {env.n_states}")
    print(f"Actions     : {env.n_actions}  {env.action_names}")
    print(f"Goal state  : {env.goal_state}")
    print(f"Obstacles   : {env.obstacle_states}")
    print(f"Slip prob   : {env.slip_prob}")
    print()

    # Show reward table
    print("Reward Table:")
    for s, r in env.rewards.items():
        row, col = env._state_to_rc(s)
        print(f"  State {s:2d} (row={row}, col={col}) → reward = {r}")

    print()

    # Run one sample episode
    state = env.reset()
    total_reward = 0
    print("Sample episode (random policy):")
    for step in range(20):
        action = np.random.choice(env.n_actions)
        next_state, reward, done = env.step(action)
        print(f"  Step {step+1:2d}: s={state:2d} → a={env.action_names[action]:5s} "
              f"→ s'={next_state:2d}  reward={reward:+d}")
        total_reward += reward
        state = next_state
        if done:
            print(f"  Episode done! Total reward: {total_reward}")
            break

    # Render grid
    env.render(title="Grid World MDP — Obstacles at S5, S10")
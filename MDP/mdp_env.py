# mdp_env.py
import numpy as np

class GridMDP:
    def __init__(self, grid_size=4, gamma=0.9):
        self.grid_size  = grid_size
        self.n_states   = grid_size * grid_size
        self.n_actions  = 4
        self.gamma      = gamma
        self.actions    = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
        self.goal_state  = self.n_states - 1
        self.start_state = 0

        # ── YOUR CONTRIBUTION: obstacles ──────────────────
        self.obstacles = [5, 7, 9, 11]   # blocked cells
        # ──────────────────────────────────────────────────

    def reward(self, state, next_state):
        if next_state == self.goal_state:
            return +10.0
        elif next_state in self.obstacles:
            return -5.0                   # big penalty for hitting wall
        elif next_state == state:
            return -1.0                   # hit boundary
        else:
            return -0.1                   # normal step cost

    def step(self, state, action):
        row, col = divmod(state, self.grid_size)
        dr, dc   = self.actions[action]
        nr = np.clip(row + dr, 0, self.grid_size - 1)
        nc = np.clip(col + dc, 0, self.grid_size - 1)
        next_state = nr * self.grid_size + nc

        # ── bounce back if obstacle ───────────────────────
        if next_state in self.obstacles:
            next_state = state

        r    = self.reward(state, next_state)
        done = (next_state == self.goal_state)
        return next_state, r, done

    def reset(self):
        return self.start_state

if __name__ == "__main__":
    env = GridMDP()
    print(f"Grid size : {env.grid_size}x{env.grid_size}")
    print(f"Obstacles : {env.obstacles}")
    print(f"Goal state: {env.goal_state}")

    s = env.reset()
    print(f"\nStart state: {s}")
    s2, r, done = env.step(s, 1)
    print(f"After DOWN → state: {s2}, reward: {r}, done: {done}")
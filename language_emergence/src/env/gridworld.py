import math
import torch


def state_int_to_tuple(state_int, grid_dim, device):
    """Convert integer state (0 to grid_dim^2 - 1) to a centered 2D coordinate tensor.
    e.g. state 0 on a 4x4 grid -> tensor([[-1.5, -1.5]])
    Returns None if state_int is None (terminal state).
    """
    if state_int is None:
        return None
    cval = (grid_dim - 1) / 2
    sx = state_int % grid_dim - cval
    sy = math.floor(state_int / grid_dim) - cval
    return torch.tensor([[sx, sy]], device=device)


def state_tuple_to_int(state, grid_dim):
    """Inverse of state_int_to_tuple. Returns None if state is None."""
    if state is None:
        return None
    sx, sy = state
    cval = (grid_dim - 1) / 2
    return int((sx + cval) + grid_dim * (sy + cval))


class SquareGridworld:
    def __init__(self, init_state, goal_state, wall_states, config):
        self.init_state = init_state
        self.goal_state = goal_state
        self.wall_states = wall_states
        self.n_states = config.grid_dim ** 2
        self.lava = config.lava
        self.grid_dim = config.grid_dim
        self.n_actions = config.n_actions
        self.step_reward = config.step_reward
        self.goal_reward = config.goal_reward
        self.wall_reward = config.wall_reward

    def get_outcome(self, state, action):
        """Return (next_state, reward) for a given state-action pair.
        next_state is None if the goal is reached (terminal).
        """
        if state == self.goal_state:
            return None, self.goal_reward

        next_state_dict = {
            0: state + 1,
            1: state + self.grid_dim,
            2: state - 1,
            3: state - self.grid_dim
        }
        cross_boundary_dict = {
            0: state % self.grid_dim == self.grid_dim - 1,
            1: state >= self.grid_dim * (self.grid_dim - 1),
            2: state % self.grid_dim == 0,
            3: state < self.grid_dim
        }

        if self.lava:
            reward = self.step_reward
            next_state = next_state_dict[action]
            if cross_boundary_dict[action]:
                reward += self.wall_reward
                next_state = state
            elif next_state in self.wall_states or state in self.wall_states:
                reward += self.wall_reward
        else:
            reward = self.step_reward
            next_state = next_state_dict[action]
            if next_state in self.wall_states or cross_boundary_dict[action]:
                next_state = state
                reward += self.wall_reward

        return int(next_state), reward

    def get_outcomes(self):
        """Return a dict mapping (state, action) -> (next_state, reward) for all pairs."""
        return {
            (s, a): self.get_outcome(s, a)
            for s in range(self.n_states)
            for a in range(self.n_actions)
        }

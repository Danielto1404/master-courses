import numpy as np
import torch
import typing as tp


class ReplayMemory:
    def __init__(
            self,
            capacity: int,
            batch_size: int,
            state_size: int,
            device: torch.device = torch.device("cpu")
    ):
        self.capacity = capacity
        self.state_size = state_size
        self.batch_size = batch_size
        self.memory = np.empty((capacity, state_size * 2 + 3), dtype=np.float32)
        self.device = device
        self.position = 0

    def push(self, state, next_state, action, reward, done):
        """Saves a transition."""
        transition = np.hstack((state, next_state, action, reward, done))
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a random batch of transitions from memory."""
        indices = np.random.choice(self.capacity, self.batch_size)
        batch = self.memory[indices]
        state = torch.from_numpy(batch[:, :self.state_size]).to(self.device)
        next_state = torch.from_numpy(batch[:, self.state_size: 2 * self.state_size]).to(self.device)
        action = torch.from_numpy(batch[:, 2 * self.state_size]).to(self.device)
        reward = torch.from_numpy(batch[:, 2 * self.state_size + 1]).to(self.device)
        done = torch.from_numpy(batch[:, 2 * self.state_size + 2]).to(self.device)
        return state, next_state, action, reward, done

    def is_ready(self):
        """Returns True if memory is ready for sampling."""
        return self.position >= self.batch_size

    def __len__(self):
        return len(self.memory)


__all__ = [
    "ReplayMemory"
]

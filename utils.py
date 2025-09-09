from typing import Deque, Tuple, Any
from collections import deque
import torch.nn as nn
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: Deque[Tuple[Any, Any, float, Any, bool]] = deque([], maxlen=capacity)

    def push(self, record: Tuple[Any, Any, float, Any, bool]) -> None:
        self.buffer.append(record)

    def pop(self) -> None:
        if len(self.buffer) > 0:
            self.buffer.pop()

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, states_, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(states_),
                np.array(dones, dtype=np.uint8))

    def __len__(self) -> int:
        return len(self.buffer)

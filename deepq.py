from typing import Deque, Tuple, Any
import torch.nn.functional as F
from collections import deque
import torch.optim as optim
import gymnasium as gym 
import torch.nn as nn
import numpy as np
import random
import torch

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(state_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.fc3: nn.Linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  

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

class DQNAgent:
    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 batch_size: int = 64,
                 mem_size: int = 100_000,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 1e-3,
                 epsilon_minimum: float = 0.01,
                 epsilon_decay_strategy: str = "linear",
                 discount_rate_gamma: float = 0.99, 
                 learning_rate_alpha: float = 1e-3,
                 update_target_every: int = 1000,
                 ) -> None:
        # tunable
        self.discount_rate_gamma: float = discount_rate_gamma
        self.learning_rate_alpha: float = learning_rate_alpha
        
        # info on environment
        self.state_dim: int = state_dim
        self.n_actions: int = n_actions

        # memory retrieve size
        self.batch_size: int = batch_size
        self.max_memory_size: int = mem_size

        # exploration/exploitation
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_minimum: float = epsilon_minimum
        self.epsilon_decay_strategy = epsilon_decay_strategy

        # to update the target
        self.steps_taken: int = 0
        self.update_target_every: int = update_target_every
    
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory: ReplayBuffer = ReplayBuffer(self.max_memory_size)

        # both nets
        self.policy_net: QNetwork = QNetwork(self.state_dim, n_actions).to(self.device)
        self.target_net: QNetwork = QNetwork(self.state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer: optim.Optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate_alpha)
        self.loss: nn.Module = nn.SmoothL1Loss()

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() > self.epsilon:
            state_tensor: torch.Tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                actions: torch.Tensor = self.policy_net(state_tensor)
                return int(torch.argmax(actions).item())
        else:
            return int(np.random.choice(range(self.n_actions)))

    def replace_network(self) -> None:
        if self.steps_taken % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def reduce_exploration(self) -> None:  # just linear per the moment
        if self.epsilon_decay_strategy == "linear":
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_minimum)
        else:
            self.epsilon = max(self.epsilon * (1 - self.epsilon_decay), self.epsilon_minimum)

    def store_transition(self, state: np.ndarray, action: int, reward: float, state_: np.ndarray, terminated: bool) -> None:
        self.memory.push((state, action, reward, state_, terminated))

        if len(self.memory) > self.max_memory_size:
            self.memory.pop()

    def sample_memory(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)

        states_tensor: torch.Tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor: torch.Tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor: torch.Tensor = torch.tensor(rewards, dtype=torch.float32)
        states__tensor: torch.Tensor = torch.tensor(states_, dtype=torch.float32)
        dones_tensor: torch.Tensor = torch.tensor(dones, dtype=torch.bool)

        return states_tensor, actions_tensor, rewards_tensor, states__tensor, dones_tensor

    def learn(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.sample_memory()
        states, actions, rewards, states_, dones = (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            states_.to(self.device),
            dones.to(self.device).float()
        )

        q_pred: torch.Tensor = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next: torch.Tensor = self.target_net(states_).max(1)[0]
            q_target: torch.Tensor = rewards + self.discount_rate_gamma * q_next * (1.0 - dones)

        loss: torch.Tensor = self.loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_taken += 1
        self.replace_network()
        self.reduce_exploration()

from torch.utils.tensorboard import SummaryWriter
from misc.utils import PrioritizedReplayBuffer, DuelingQNetwork
from dataclasses import dataclass
import torch.optim as optim
from typing import Tuple
import torch.nn as nn
import numpy as np
import torch
import os

@dataclass
class DQNConfig():
    state_dim: int
    n_actions: int
    batch_size: int = 64
    mem_size: int = 100_000
    epsilon: float = 1.0
    epsilon_decay: float = 1e-3
    epsilon_min: float = 0.01
    epsilon_decay_strategy: str = "linear"
    dr_gamma: float = 0.99
    lr_alpha: float = 1e-3
    tau: float = 0.01
    checkpoint_path: str = "dqn_checkpoint.pth"
    run_dir: str = "runs/dqn_experiment"
    load: bool = False

class DuelingDDQNAgentPER():
    def __init__(self, config: DQNConfig) -> None:
        self.config = config
        # learning sensitivity
        self.dr_gamma: float = config.dr_gamma
        self.lr_alpha: float = config.lr_alpha
        
        # info on environment
        self.state_dim: int = config.state_dim
        self.n_actions: int = config.n_actions

        # memory retrieve size
        self.batch_size: int = config.batch_size
        self.max_memory_size: int = config.mem_size

        # exploration/exploitation
        self.epsilon: float = config.epsilon
        self.epsilon_decay: float = config.epsilon_decay
        self.epsilon_min: float = config.epsilon_min
        self.epsilon_decay_strategy = config.epsilon_decay_strategy

        # to update the target
        self.steps_taken: int = 0
        self.tau: float = config.tau
    
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory: PrioritizedReplayBuffer = PrioritizedReplayBuffer(self.max_memory_size)

        self.policy_net: DuelingQNetwork= DuelingQNetwork(self.state_dim, config.n_actions).to(self.device)
        self.target_net: DuelingQNetwork = DuelingQNetwork(self.state_dim, config.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer: optim.Optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr_alpha, amsgrad=True)

        self.writer = SummaryWriter(log_dir=config.run_dir)

        if config.load:
            self.load_checkpoint(f'{config.checkpoint_path}')

        #after reading grokking deep reinforcement learning I understood that the loss doesn't need to punish high errors like MSE
        self.loss: nn.Module = nn.HuberLoss(reduction="none")

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() > self.epsilon:
            state_tensor: torch.Tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                actions: torch.Tensor = self.policy_net(state_tensor)
                return int(torch.argmax(actions).item())
        else:
            return int(np.random.choice(range(self.n_actions)))

    def replace_network(self) -> None:
        #soft update in the replacement of the target and policy network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def reduce_exploration(self) -> None:  
        if self.epsilon_decay_strategy == "linear":
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        else:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.writer.add_scalar("Policy/epsilon", self.epsilon, self.steps_taken)

    def store_transition(self, state: np.ndarray, action: int, reward: float, state_: np.ndarray, terminated: bool) -> None:
        self.memory.push((state, action, reward, state_, terminated))

    def sample_memory(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, states_, dones, indices, weights = self.memory.sample(self.batch_size)

        states_tensor:  torch.Tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor: torch.Tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor: torch.Tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        states__tensor: torch.Tensor = torch.tensor(states_, dtype=torch.float32, device=self.device)
        dones_tensor:   torch.Tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        indices_tensor: torch.Tensor = torch.tensor(indices, dtype=torch.float32, device=self.device)
        weights_tensor: torch.Tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return states_tensor, actions_tensor, rewards_tensor, states__tensor, dones_tensor, indices_tensor, weights_tensor

    def save_checkpoint(self, path):
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_taken': self.steps_taken,
        }

        torch.save(checkpoint, path)
        print("Checkpoint saved.")

    def load_checkpoint(self, path):
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.steps_taken = checkpoint['steps_taken']
                print(f"Checkpoint '{path}' loaded successfully!")
            except (EOFError, RuntimeError, KeyError) as e:
                print(f"Failed to load checkpoint '{path}': {e}")
                print("Starting fresh.")
        else:
            print(f"No checkpoint found at '{path}'. Starting fresh.")

    def learn(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, states_, dones, indices, weights = self.sample_memory()

        q_pred: torch.Tensor = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(states_), dim=-1)
            q_next: torch.Tensor = self.target_net(states_).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target: torch.Tensor = rewards + self.dr_gamma * q_next * (1.0 - dones)

        td_errors = (q_target - q_pred).detach()

        loss: torch.Tensor = (self.loss(q_pred, q_target) * weights).mean()
        self.writer.add_scalar("Loss/train", loss.item(), self.steps_taken)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_errors_numpy = td_errors.cpu().numpy()
        self.memory.update(indices, td_errors_numpy)

        self.steps_taken += 1
        self.replace_network()
    




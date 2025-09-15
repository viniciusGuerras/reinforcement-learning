from typing import Deque, Tuple, Any
import torch.nn.functional as F
from collections import deque
import torch.nn as nn
import numpy as np
import random
import torch
import math

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

#https://arxiv.org/abs/1511.06581
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, adv_type="avg", fc1=128, fc2=64, fc3=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc_adv = nn.Linear(fc2, fc3)
        self.fc_value  = nn.Linear(fc2, fc3)
        self.adv   = nn.Linear(fc3, action_dim)
        self.value = nn.Linear(fc3, 1)
        self.adv_type = adv_type

    def forward(self, x):
        features = F.relu(self.fc2(F.relu(self.fc1(x))))
        x_adv = self.adv(F.relu(self.fc_adv(features)))
        x_value = self.value(F.relu(self.fc_value(features)))
        if self.adv_type == 'avg':
          advAverage = torch.mean(x_adv, dim=1, keepdim=True)
          q =  x_value + x_adv - advAverage
        else:
          advMax,_ = torch.max(x_adv, dim=1, keepdim=True)
          q =  x_value + x_adv - advMax
        return q

#https://arxiv.org/abs/1706.10295
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.sigma_zero = sigma_zero
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_mu      = nn.Parameter(torch.empty(out_features))
            self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)

        self.register_buffer("epsilon_input", torch.zeros(in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features))

        #cool activation
        mu_range = 1 / (in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)

        if bias:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_zero / (in_features ** 0.5))

        self.weight_sigma.data.fill_(self.sigma_zero / (in_features ** 0.5))

    def _f(self, x):
        return x.sign() * x.abs().sqrt()

    def _sample_noise(self):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        eps_in = self._f(self.epsilon_input)
        eps_out = self._f(self.epsilon_output)

        weight_epsilon = eps_out.outer(eps_in)  # outer product
        bias_epsilon = eps_out
        return weight_epsilon, bias_epsilon

    def forward(self, x):
        weight_epsilon, bias_epsilon = self._sample_noise()
        weight = self.weight_mu + self.weight_sigma * weight_epsilon
        if self.bias_mu is not None:
            bias = self.bias_mu + self.bias_sigma * bias_epsilon
        else:
            bias = None
        return F.linear(x, weight, bias)

class FactorizedNoisyDuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, sigma_zero=0.4):
        super().__init__()
        # shared feature extractor (one FC layer here — you can expand)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # advantage stream
        self.adv_fc = nn.Linear(hidden_dim, hidden_dim)
        self.noisy_adv = NoisyLinear(hidden_dim, action_dim, sigma_zero=sigma_zero)
        # value stream
        self.val_fc = nn.Linear(hidden_dim, hidden_dim)
        self.noisy_val = NoisyLinear(hidden_dim, 1, sigma_zero=sigma_zero)

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        adv_h = F.relu(self.adv_fc(feat))
        val_h = F.relu(self.val_fc(feat))
        adv = self.noisy_adv(adv_h)   # shape (batch, action_dim)
        val = self.noisy_val(val_h)   # shape (batch, 1)
        adv_mean = adv.mean(dim=1, keepdim=True)
        q = val + adv - adv_mean
        return q

class ConvolutionalQNetwork(nn.Module):
    def __init__(self, input_shape, action_dim, hidden_dim=128):
        super().__init__()
        self.convNet = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.net = nn.Sequential(
            nn.Linear(self._get_conv_output(input_shape), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def _get_conv_output(self, shape):
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self.convNet(dummy_input)
        return int(torch.prod(torch.tensor(dummy_output.size())))

    def forward(self, x):
        x = self.convNet(x).view(x.size()[0], -1)
        return self.net(x)

class ReplayBuffer():
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

class SumTree():
    def __init__(self, capacity):
        self.capacity = capacity 
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_pointer = 0
        self.max_priority = 0

    def add(self, priority):
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.max_priority = max(self.max_priority, priority)
        return tree_idx

    def update(self, tree_idx, priority):
        tree_idx = int(tree_idx)
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
        self.max_priority = max(self.max_priority, priority)
        
    def get_leaf(self, v): 
        parent_idx = 0 
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):  
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child]:
                    parent_idx = left_child
                else:
                    v -= self.tree[left_child]
                    parent_idx = right_child
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    def total_priority(self):
        return self.tree[0]  

#https://arxiv.org/abs/1511.05952
class PrioritizedReplayBuffer(): 
    def __init__(self, capacity: int = 100_000, alpha = 0.6, beta = 0.4, beta_increase=1e-6, rank_based=False) -> None:
        self.capacity = capacity
        self.alpha = alpha    #dictates the probability  
        self.beta = beta      #dictates the strength of the importance sampling
        self.beta_increase = beta_increase
        self.rank_based = rank_based
        self.epsilon = 1e-6

        self.data = [None] * capacity
        self.priorities = SumTree(capacity)
        self.size = 0

    def push(self, record: Tuple[Any, Any, float, Any, bool]) -> None:
        priority = self.priorities.max_priority if self.size > 0 else 1.0
        tree_idx = self.priorities.add(priority)
        self.data[tree_idx - self.capacity + 1] = record
        self.size = min(self.size + 1, self.capacity)

    def update(self, indices, td_errors):
        if self.rank_based: 
            abs_errors = np.abs(td_errors) + self.epsilon
            sorted_indices = np.argsort(-abs_errors)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(1, len(td_errors) + 1)
                
            priorities = 1 / (ranks ** self.alpha)

            for idx, priority in zip(indices, priorities):
                idx = int(idx)
                self.priorities.update(idx, priority) 
                self.priorities.max_priority = max(self.priorities.max_priority, priority)
        else:
            for idx, td_error in zip(indices, td_errors):
                idx = int(idx)
                priority = (abs(td_error) + self.epsilon) ** self.alpha
                self.priorities.update(idx, priority) 
                self.priorities.max_priority = max(self.priorities.max_priority, priority)

    def beta_anneal(self):
        self.beta = min(self.beta + self.beta_increase, 1)

    def pop(self) -> None:
        if len(self.data) > 0:
            self.data.pop()
            self.priorities.pop()

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch, indices, priorities = [], [], []
        total = self.priorities.total_priority()

        segments = total/batch_size
        for i in range(batch_size):
            a, b = i * segments, (i + 1) * segments
            num = np.random.uniform(a, b)  
            tree_idx, priority, data_idx =self.priorities.get_leaf(num)

            while self.data[data_idx] is None:
                num = np.random.uniform(a, b)
                tree_idx, priority, data_idx = self.priorities.get_leaf(num)

            batch.append(self.data[data_idx])
            indices.append(tree_idx)
            priorities.append(priority)

        states, actions, rewards, next_states, dones = zip(*batch)
        sampling_probs = np.array(priorities) / self.priorities.total_priority()
        weights = (self.size * sampling_probs) ** -self.beta
        weights /= weights.max()

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
            np.array(indices),
            np.array(weights, dtype=np.float32)
        )

    def __len__(self) -> int:
        return int(self.size)
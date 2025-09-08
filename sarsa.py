import gymnasium as gym 
from checkpointer import Cornifer
import numpy as np

class SARSA():
    def __init__(self, env: gym.Env, epsilon: float = 1, epsilon_decay: float = 0.0005, 
                 epsilon_minimum: float = 0.01, epsilon_decay_strategy: str = "linear", 
                 learning_rate_alpha: float = 0.5, discount_rate_gamma: float = 0.9, 
                 checkpoint: bool = True, checkpoint_spacing: int = 1000, filename: str | None = None):
        self.epsilon_decay_strategy: str = epsilon_decay_strategy
        self.learning_rate_alpha: float = learning_rate_alpha
        self.discount_rate_gamma: float = discount_rate_gamma
        self.checkpoint_spacing: int = checkpoint_spacing
        self.epsilon_minimum: float = epsilon_minimum
        self.epsilon_decay: float = epsilon_decay  
        self.episode_rewards: list[float] = []
        self.filename: str | None = filename
        self.checkpoint: bool = checkpoint
        self.epsilon: float = epsilon
        self.env: gym.Env = env

        q_table = Cornifer.load(filename=filename, suffix="sarsa")
        if q_table is None:
            self.q_table: np.ndarray = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        else:
            self.q_table: np.ndarray = q_table

    def act(self, state: int) -> int:
        if np.random.rand() > self.epsilon:
            return int(np.argmax(self.q_table[state]))
        else:
            return int(self.env.action_space.sample())

    def act_greedy(self, state: int) -> int:
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, state_: int, action: int, action_: int, reward: float) -> None:
        """Bellman equation for SARSA."""
        self.q_table[state, action] += self.learning_rate_alpha * (
            reward + (self.discount_rate_gamma * self.q_table[state_, action_] - self.q_table[state, action])
        ) 

    def exploration_reduction(self) -> None:
        if self.epsilon_decay_strategy == "linear":
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_minimum)
        elif self.epsilon_decay_strategy == "exponential":
            self.epsilon = max(self.epsilon * (1 - self.epsilon_decay), self.epsilon_minimum)

    def train(self, epochs: int = 1000) -> list[float]:
        for epoch in range(epochs):
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                #get the action from the model
                action = self.act(state)

                #make the action in the environment and get the results
                state_, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                #get the current optimal action in the resulting state
                action_ = self.act(state_)

                #update the model via bellman equation
                self.update(state, state_, action, action_, reward)
                state = state_
            self.exploration_reduction()
            self.episode_rewards.append(total_reward)

            if self.checkpoint and (epoch % self.checkpoint_spacing) == 0:
                Cornifer.checkpoint(self.q_table, self.filename, epoch, suffix="sarsa")
        Cornifer.save(self.q_table, self.filename, suffix="sarsa")
        return self.episode_rewards
    
    def test(self, times: int = 10) -> list[float]:
        temp_rewards: list[float] = []
        for _ in range(times):
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.act_greedy(state)
                state_, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = state_
            temp_rewards.append(total_reward)
        return temp_rewards

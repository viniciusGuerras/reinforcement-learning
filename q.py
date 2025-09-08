from checkpointer import Cornifer
import gymnasium as gym 
import numpy as np

class QLearning():
    def __init__(self, env, epsilon=1, epsilon_decay=0.0005, epsilon_decay_strategy="linear", learning_rate_alpha=0.5, discount_rate_gamma=0.9, checkpoint=True, checkpoint_spacing=1000, filename=None):
        self.env = env
        self.epsilon = epsilon
        self.episode_rewards = []
        self.epsilon_decay = epsilon_decay  
        self.epsilon_decay_strategy = epsilon_decay_strategy
        self.learning_rate_alpha = learning_rate_alpha
        self.discount_rate_gamma = discount_rate_gamma
        self.checkpoint = checkpoint
        self.checkpoint_spacing = checkpoint_spacing
        self.filename = filename

        q_table = Cornifer.load(filename=filename, suffix="q")
        if q_table is None:
            self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        else:
            self.q_table = q_table

    def act(self, state):
        if np.random.rand() > self.epsilon:
                return np.argmax(self.q_table[state])
        else:
            return self.env.action_space.sample()

    def act_greedy(self, state):
        return np.argmax(self.q_table[state])

    def update(self, state, state_, action, reward): #bellman equation for q-learning
        self.q_table[state, action] += self.learning_rate_alpha * (
            reward + (self.discount_rate_gamma * np.max(self.q_table[state_]) - self.q_table[state, action])
            ) 

    def exploration_reduction(self, min=0.00):
        if self.epsilon_decay_strategy == "linear":
            self.epsilon = max(self.epsilon - self.epsilon_decay, min)
        elif self.epsilon_decay_strategy == "exponential":
            self.epsilon = max(self.epsilon * (1 - self.epsilon_decay), min)

    def train(self, epochs=1000):
        for epoch in range(epochs):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.act(state)
                state_, reward, terminated, truncated, info  = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                self.update(state, state_, action, reward)
                state = state_
            self.exploration_reduction(min=0.001)
            self.episode_rewards.append(total_reward)

            if self.checkpoint and (epoch % self.checkpoint_spacing) == 0:
                Cornifer.checkpoint(self.q_table, self.filename, epoch, suffix="q")
        Cornifer.save(self.q_table, self.filename, "q")
        return self.episode_rewards
    
    def test(self, times=10):
        temp_rewards = []
        for i in range(times):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.act_greedy(state)
                state_, reward, terminated, truncated, info  = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = state_
            temp_rewards.append(total_reward)
        return temp_rewards
                




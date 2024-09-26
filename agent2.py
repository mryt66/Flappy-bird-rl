import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from model import DQN

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1])
        q_values = self.model(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.tensor([x[0] for x in minibatch], dtype=torch.float32)
        actions = torch.tensor([x[1] for x in minibatch])
        rewards = torch.tensor([x[2] for x in minibatch], dtype=torch.float32)
        next_states = torch.tensor([x[3] for x in minibatch], dtype=torch.float32)
        dones = torch.tensor([x[4] for x in minibatch], dtype=torch.float32)
        
        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = torch.max(next_q_values, dim=1)[0]
        
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        
        loss = self.loss_fn(q_value, expected_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
        
        # Update target network every 10 iterations
        if len(self.memory) % 10 == 0:
            self.target_model.load_state_dict(self.model.state_dict())


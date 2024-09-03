import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from model import DQN

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.98, buffer_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        
        self.model = DQN(state_dim, action_dim).to('cuda')
        self.target_model = DQN(state_dim, action_dim).to('cuda')
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1])
        
        # Convert NumPy array to PyTorch tensor and move to CUDA
        state_tensor = torch.tensor(state, dtype=torch.float32).to('cuda')
        
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    
    def remember(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float32).to('cuda')
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to('cuda')
        self.memory.append((state_tensor, action, reward, next_state_tensor, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, batch_size))
        
        states = torch.stack(states, dim=0).to('cuda')
        actions = torch.tensor(actions).to('cuda')
        rewards = torch.tensor(rewards, dtype=torch.float32).to('cuda')
        next_states = torch.stack(next_states, dim=0).to('cuda')
        dones = torch.tensor(dones, dtype=torch.float32).to('cuda')
            
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
        
        if len(self.memory) % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))

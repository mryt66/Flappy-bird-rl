import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from model import DQN
from parameters import (
    DEVICE,
    lr,
    gamma,
    epsilon,
    epsilon_min,
    epsilon_decay,
    buffer_size,
    batch_size,
    target_update,
    state_dim,
    action_dim,
)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr,
        gamma,
        epsilon,
        epsilon_decay,
        epsilon_min,
        buffer_size,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.steps_done = 0

    def remember(self, state, action, reward, next_state, done):
        if len(state) != self.state_dim or len(next_state) != self.state_dim:
            print("State shape mismatch: ", state, next_state)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # print(state)
        
        if random.random() < self.epsilon:
            return random.choices([0, 1], weights=[0.70, 0.30])[0]
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

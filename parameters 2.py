#parameters.py
import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
lr = 0.0005        # Learning rate
gamma = 0.99        # Discount factor
epsilon = 1.0       # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.998  # Decay rate for exploration
buffer_size = 100000     # Replay buffer size
batch_size = 64        # Mini-batch size for training
target_update = 5      # How often to update the target network
penalty = -10          # Penalty for losing the game

# Other parameters
state_dim = 4    # Number of state dimensions
action_dim = 2   # Number of actions (jump or do nothing)
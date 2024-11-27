import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#training parameters
lr = 0.0005
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9975
buffer_size = 100000
batch_size = 64
target_update = 5
penalty = -10

state_dim = 4
action_dim = 2
patience = 400
min_improvement = 0.01  

#game parameters
RECT_WIDTH = 35
RECT_HEIGHT = 35
PIPE_WIDTH = 40
GRAVITY = 1
PIPE_GAP = 140

JUMP_STRENGTH = -11
PIPE_SPEED = 6

#traiing with bigger model in model.py
# RECT_WIDTH = 35
# RECT_HEIGHT = 35
# PIPE_WIDTH = 70
# GRAVITY = 1
# PIPE_GAP = 150
# JUMP_STRENGTH = -9
# PIPE_SPEED = 6

#new_model training
# RECT_WIDTH = 35
# RECT_HEIGHT = 35
# PIPE_WIDTH = 70
# PIPE_GAP = 180
# GRAVITY = 1
# JUMP_STRENGTH = -8
# PIPE_SPEED = 6

#best.pth
# RECT_WIDTH = 35
# RECT_HEIGHT = 35
# PIPE_WIDTH = 70
# PIPE_GAP = 220
# GRAVITY = 1
# JUMP_STRENGTH = -10
# PIPE_SPEED = 5


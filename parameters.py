import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#training parameters
lr = 0.0005
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.998
buffer_size = 100000
batch_size = 64
target_update = 5
penalty = -10

state_dim = 4
action_dim = 2

#game parameters
RECT_WIDTH = 35
RECT_HEIGHT = 35
PIPE_WIDTH = 70
PIPE_GAP = 180
GRAVITY = 1
JUMP_STRENGTH = -12
PIPE_SPEED = 6

#best.pth
# RECT_WIDTH = 35
# RECT_HEIGHT = 35
# PIPE_WIDTH = 70
# PIPE_GAP = 220
# GRAVITY = 1
# JUMP_STRENGTH = -10
# PIPE_SPEED = 5


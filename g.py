import pygame
import random
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent2 import DQNAgent
from parameters import lr, gamma, epsilon, epsilon_decay, buffer_size, penalty

# Game settings
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

# Color definitions
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Object dimensions and physics settings
RECT_WIDTH = 35
RECT_HEIGHT = 35
PIPE_WIDTH = 70
PIPE_GAP = 220
GRAVITY = 1
JUMP_STRENGTH = -13
PIPE_SPEED = 5

class Rectangle:
    def __init__(self):
        self.x = SCREEN_WIDTH // 4
        self.y = SCREEN_HEIGHT // 2
        self.y_speed = 0

    def distance(self, pipes):
        for pipe in pipes:
            if pipe.top.left > self.x:
                dist_horizontal = pipe.top.left - self.x
                gap_y_center = pipe.top.height + PIPE_GAP // 2
                dist_vertically = self.y + RECT_HEIGHT // 2 - gap_y_center
                return dist_horizontal, dist_vertically
        return None, None

    def jump(self):
        self.y_speed = JUMP_STRENGTH

    def apply_gravity(self):
        self.y_speed += GRAVITY
        self.y += self.y_speed

class Pipe:
    def __init__(self):
        self.top = pygame.Rect(SCREEN_WIDTH, 0, PIPE_WIDTH, random.randint(15, SCREEN_HEIGHT - PIPE_GAP - 15))
        self.bottom = pygame.Rect(SCREEN_WIDTH, self.top.height + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT - self.top.height - PIPE_GAP)

    def move(self):
        self.top.x -= PIPE_SPEED
        self.bottom.x -= PIPE_SPEED

    def off_screen(self):
        return self.top.right < 0

def get_observation():
    dist_horizontal, dist_vertically = rectangle.distance(pipes)
    if dist_horizontal is None or dist_vertically is None:
        dist_horizontal = SCREEN_WIDTH
        dist_vertically = SCREEN_HEIGHT // 2
    
    normalized_horizontal = dist_horizontal / SCREEN_WIDTH
    normalized_vertical = dist_vertically / SCREEN_HEIGHT
    rect_y_normalized = rectangle.y / SCREEN_HEIGHT
    rect_y_speed_normalized = rectangle.y_speed / JUMP_STRENGTH
    
    observation = np.array([
        normalized_horizontal,
        normalized_vertical,
        rect_y_normalized,
        rect_y_speed_normalized
    ])
    
    return observation

def take_action(action):
    if action == 1:  # Jump
        rectangle.jump()

# Initialize the game components
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Rectangle")

rectangle = Rectangle()
pipes = []
pipe_timer = 0
score = 0
# font = pygame.font.SysFont(None, 36)
game_active = True


experience_buffer = deque(maxlen=10000)
obs = 4
actions = 2

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(state_dim=obs, action_dim=actions, lr=lr, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, buffer_size=buffer_size)

episode_rewards = []
num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100

for episode in range(num_episodes):
    rectangle.x = SCREEN_WIDTH // 4
    rectangle.y = SCREEN_HEIGHT // 2
    rectangle.y_speed = 0
    
    pipes = []
    pipe_timer = 89
    score = 0
    game_active = True
    
    episode_reward = 0
    
    while game_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Get current observation
        obs = get_observation()

        # Choose an action
        action = agent.act(obs)

        # Take the action
        take_action(action)

        rectangle.apply_gravity()

        pipe_timer += 1
        if pipe_timer > 90:
            pipes.append(Pipe())
            pipe_timer = 0

        for pipe in pipes:
            pipe.move()
            if pipe.off_screen():
                pipes.remove(pipe)
                score += 1
                # reward = 1  # Adjust the reward as needed

        # Check for collisions
        done = False
        reward = 15
        for pipe in pipes:
            if pygame.Rect(rectangle.x, rectangle.y, RECT_WIDTH, RECT_HEIGHT).colliderect(pipe.top) or \
               pygame.Rect(rectangle.x, rectangle.y, RECT_WIDTH, RECT_HEIGHT).colliderect(pipe.bottom):
                game_active = False
                done = True
                reward = -1000
                break
        if rectangle.y <= 0 or rectangle.y + RECT_HEIGHT >= SCREEN_HEIGHT:
            game_active = False
            done = True
            reward = -1000

        next_obs = get_observation()
        
        agent.remember(obs, action, reward, next_obs, done)
        episode_reward += reward
        
        # Update the agent
        agent.replay(batch_size=128)

    episode_rewards.append(episode_reward)
    print(f"Episode {episode+1}, Reward: {episode_reward}, {score}")

# Save the trained model
if not os.path.exists('models/'):
    os.makedirs('models/')
number = len(os.listdir('models/')) + 1
torch.save(agent.model.state_dict(), f"models/flappy{number}.pth")

# Plot the rewards
plt.figure(figsize=(12, 6))
plt.plot(range(len(episode_rewards)), episode_rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

pygame.quit()

import pygame
import random
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent_cuda import DQNAgent
import time

pygame.init()

# Game settings
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

# Color definitions
RECT_WIDTH = 35
RECT_HEIGHT = 35
PIPE_WIDTH = 70
PIPE_GAP = 220
GRAVITY = 1
JUMP_STRENGTH = -13
PIPE_SPEED = 5

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class Rectangle:
    def __init__(self):
        self.x = SCREEN_WIDTH // 4
        self.y = SCREEN_HEIGHT // 2
        self.y_speed = 0

    def draw(self, screen):
        pygame.draw.rect(screen, RED, (self.x, self.y, RECT_WIDTH, RECT_HEIGHT))

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

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.top)
        pygame.draw.rect(screen, WHITE, self.bottom)

    def move(self):
        self.top.x -= PIPE_SPEED
        self.bottom.x -= PIPE_SPEED

    def off_screen(self):
        return self.top.right < 0

def get_observation(rectangle, pipes):
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

def take_action(rectangle, action):
    if action == 1:  # Jump
        rectangle.jump()

if not os.path.exists('models/'):
    os.makedirs('models/')
if not os.path.exists(f"models/Training_{len(os.listdir('models/'))+1}/"):
    os.makedirs(f"models/Training_{len(os.listdir('models/'))+1}/")

# Initialize pygame components (no screen needed for faster training)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

reward = 0.1
# Parameters
num_agents = 5
num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100

obs = 4 # distances 2xverticaly, horizantally, position
actions = 2 # space or do nothing

lr = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
buffer_size = 50000
penalty = -1

agents = [DQNAgent(state_dim=obs, action_dim=actions, lr=lr, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, buffer_size=buffer_size) for _ in range(num_agents)]

episode_rewards = [[] for _ in range(num_agents)]

copy_frequency = 3

for episode in range(num_episodes):
    start_time = time.time()
    rectangles = [Rectangle() for _ in range(num_agents)]
    pipe_timer = 89
    pipes = []
    game_active = [True] * len(agents)
    agent_reward = [[_, 0] for _ in range(num_agents)]

    while any(game_active):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.fill(BLACK)
        
        reward = 0.1
        pipe_timer += 1
        if pipe_timer > 90:
            pipes.append(Pipe())
            pipe_timer = 0
        for pipe in pipes:
            pipe.move()
            if pipe.off_screen():
                pipes.remove(pipe)
                reward += 5
        for pipe in pipes:
            pipe.draw(screen)
        
        for i, rectangle in enumerate(rectangles):
            
            if game_active[i]:
                obs = get_observation(rectangle, pipes)
                action = agents[i].act(obs)
                take_action(rectangle, action)
                rectangle.apply_gravity()
                rectangle.draw(screen)
                done = False

                for pipe in pipes:
                    if pygame.Rect(rectangle.x, rectangle.y, RECT_WIDTH, RECT_HEIGHT).colliderect(pipe.top) or \
                       pygame.Rect(rectangle.x, rectangle.y, RECT_WIDTH, RECT_HEIGHT).colliderect(pipe.bottom):
                        game_active[i] = False
                        done = True
                        reward = -1
                        break
                if rectangle.y <= 0 or rectangle.y + RECT_HEIGHT >= SCREEN_HEIGHT:
                    game_active[i] = False
                    done = True
                    reward = -1

                next_obs = get_observation(rectangle, pipes)
                agents[i].remember(obs, action, reward, next_obs, done)
                agent_reward[i][1] += reward
                
        pygame.display.flip()
        pygame.time.Clock().tick(60)
        
        for agent in agents:
            agent.replay(batch_size=32)
        
    sorted_rewards = sorted(agent_reward, key=lambda x: x[1], reverse=True)
    best_agent_index = sorted_rewards[0][0]
    best_model_state_dict = agents[best_agent_index].model.state_dict()

    if (episode + 1) % copy_frequency == 0:
        for i, agent in enumerate(agents):
            if i != best_agent_index:
                agent.model.load_state_dict(best_model_state_dict)
    
    torch.save(best_model_state_dict, f"models/Training_{len(os.listdir('models/'))}/best_{episode}.pth")
    
    average = sum([x[1] for x in sorted_rewards]) / len(sorted_rewards)
    print(f"Episode {episode+1}, Average: {average}, Time: {time.time() - start_time}")
    

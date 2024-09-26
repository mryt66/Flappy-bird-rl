import pygame
import random
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean
from collections import deque

from agent_cuda import DQNAgent
from parameters import lr, gamma, epsilon, epsilon_decay, buffer_size, live, penalty, batch_size


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

def take_action(action, rectangle):
    if action == 1:  # Jump
        rectangle.jump()

if not os.path.exists('models/'):
    os.makedirs('models/')
if not os.path.exists(f"models/Training_{len(os.listdir('models/'))+1}/"):
    os.makedirs(f"models/Training_{len(os.listdir('models/'))+1}/")

# Initialize pygame components (no screen needed for faster training)
# pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Parameters
num_agents = 10
num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 500

obs = 4 # distances 2xverticaly, horizantally, position
actions = 2 # space or do nothing


agents = [DQNAgent(state_dim=obs, action_dim=actions, lr=lr, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, buffer_size=buffer_size) for _ in range(num_agents)]

episode_rewards = [[] for _ in range(num_agents)]

average_improvement = []

for episode in range(num_episodes):
    time1 = time.time()
    rectangles = [Rectangle() for _ in range(num_agents)]
    pipes = [[] for _ in range(num_agents)]
    pipe_timers = [0 for _ in range(num_agents)]
    scores = [0 for _ in range(num_agents)]
    game_actives = [True for _ in range(num_agents)]
    
    episode_reward = [[_, 0] for _ in range(num_agents)]
    
    while any(game_actives):
        for i in range(num_agents):
            if not game_actives[i]:
                continue

            # Get current observation
            obs = get_observation(rectangles[i], pipes[i])

            # Choose an action
            action = agents[i].act(obs)

            # Take the action
            take_action(action, rectangles[i])

            rectangles[i].apply_gravity()
            
            reward = 15
            pipe_timers[i] += 1
            if pipe_timers[i] > 90:
                pipes[i].append(Pipe())
                pipe_timers[i] = 0

            for pipe in pipes[i]:
                pipe.move()
                if pipe.off_screen():
                    pipes[i].remove(pipe)
                    # reward = 5  # Adjust the reward as needed
                    scores[i] += 1

            # Check for collisions
            done = False
            
            for pipe in pipes[i]:
                if pygame.Rect(rectangles[i].x, rectangles[i].y, RECT_WIDTH, RECT_HEIGHT).colliderect(pipe.top) or \
                   pygame.Rect(rectangles[i].x, rectangles[i].y, RECT_WIDTH, RECT_HEIGHT).colliderect(pipe.bottom):
                    game_actives[i] = False
                    done = True
                    reward = penalty
                    break
            if rectangles[i].y <= 0 or rectangles[i].y + RECT_HEIGHT >= SCREEN_HEIGHT:
                game_actives[i] = False
                done = True
                reward = penalty 
            next_obs = get_observation(rectangles[i], pipes[i])
            
            agents[i].remember(obs, action, reward, next_obs, done)
            episode_reward[i][1] += reward # episode_reward=[[0,reward],[1,reward]...]
            
            # Update the agent (could be done collectively after the loop)
            agents[i].replay(batch_size=batch_size)

    #save the model
    sorted_rewards = sorted(episode_reward, key=lambda x: x[1], reverse=True)
    torch.save(agents[sorted_rewards[0][0]].model.state_dict(), f"models/Training_{len(os.listdir('models/'))}/best_{episode}.pth")
    
    average = np.average(sorted_rewards) 
    average_improvement.append(average)
    mean_improvement = np.average(average_improvement) 
    
    median = sorted_rewards[len(sorted_rewards)//2][1]
    max_scores = max(scores)
    time2 = time.time() - time1
    
    print(f"Episode_{episode}: {average}  | {sorted_rewards[0][1]} |  Avg_improvment: {mean_improvement} | {max_scores} | {time2:.2f}")
    with open(f"models/Training_{len(os.listdir('models/'))}/outputs.txt", "a") as file:
        file.write(f"{episode} | {median} | {average} | Avg_improvment: {mean_improvement} |      {max_scores}      | {time2:.2f}\n")
file.close()

pygame.quit()

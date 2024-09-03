import pygame
import random
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent_cuda import DQNAgent

pygame.init()

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

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

# Initialize the game components
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Rectangle")

agents = [DQNAgent(state_dim=4, action_dim=2, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000) for _ in range(10)]
episode_rewards = [[] for _ in range(10)]
num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100

for episode in range(num_episodes):
    rectangles = [Rectangle() for _ in agents]
    pipes = []
    pipe_timer = 0
    scores = [0] * len(agents)
    game_active = [True] * len(agents)
    
    episode_reward = [0] * len(agents)
    
    while any(game_active):
        screen.fill(BLACK)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for i, rectangle in enumerate(rectangles):
            if game_active[i]:
                obs = get_observation(rectangle, pipes)
                action = agents[i].act(obs)
                take_action(rectangle, action)
                rectangle.apply_gravity()

                pipe_timer += 1
                if pipe_timer > 90:
                    pipes.append(Pipe())
                    pipe_timer = 0

                for pipe in pipes:
                    pipe.move()
                    if pipe.off_screen():
                        pipes.remove(pipe)
                        scores[i] += 1
                        episode_reward[i] += 3  # Adjust the reward as needed

                for pipe in pipes:
                    pipe.draw(screen)

                rectangle.draw(screen)

                done = False
                reward = 0.1
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
                episode_reward[i] += reward

                # Update the agent
                agents[i].replay(batch_size=32)

        pygame.display.flip()
        pygame.time.Clock().tick(10)

    for i in range(len(agents)):
        episode_rewards[i].append(episode_reward[i])
        print(f"Episode {episode+1}, Agent {i+1} Reward: {episode_reward[i]}")

# Save the trained models
for i, agent in enumerate(agents):
    number = len(os.listdir('models/')) + 1
    torch.save(agent.model.state_dict(), f"models/flappy_agent_{i+1}_{number}.pth")

# Plot the rewards
for i, rewards in enumerate(episode_rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(rewards)), rewards)
    plt.title(f"Episode Rewards for Agent {i+1}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

pygame.quit()

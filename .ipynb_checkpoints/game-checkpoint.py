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

# from agent_cuda import DQNAgent
from agent2 import DQNAgent
from parameters import lr, gamma, epsilon, epsilon_decay, buffer_size, live, penalty, batch_size

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
        # Find the first pipe that is to the right of the rectangle
        for pipe in pipes:
            if pipe.top.left > self.x:
                # Calculate the horizontal distance to the pipe
                dist_horizontal = pipe.top.left - self.x

                # Calculate the vertical distance to the center of the gap in the pipe
                gap_y_center = pipe.top.height + PIPE_GAP // 2
                dist_vertically = self.y + RECT_HEIGHT // 2 - gap_y_center

                return dist_horizontal, dist_vertically

        return None, None  # If no pipe is to the right

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


def get_observation():
    dist_horizontal, dist_vertically = rectangle.distance(pipes)
    if dist_horizontal is None or dist_vertically is None:
        dist_horizontal = SCREEN_WIDTH
        dist_vertically = SCREEN_HEIGHT // 2
    
    # Normalize the distances
    normalized_horizontal = dist_horizontal / SCREEN_WIDTH
    normalized_vertical = dist_vertically / SCREEN_HEIGHT
    
    # Include the rectangle's y-position and velocity
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
pipe_timer = 80
score = 0
font = pygame.font.SysFont(None, 36)
game_active = True  # Start the game immediately


experience_buffer = deque(maxlen=10000)
obs = 4
actions = 2

agent = DQNAgent(state_dim=obs, action_dim=actions, lr=lr, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, buffer_size=buffer_size)

episode_rewards = []
num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 400

for episode in range(num_episodes):
    time1 = time.time()
    rectangle.x = SCREEN_WIDTH // 4
    rectangle.y = SCREEN_HEIGHT // 2
    rectangle.y_speed = 0
    
    pipes = []
    pipe_timer = 89
    score = 0
    game_active = True
    
    episode_reward = 0
    
    while game_active:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        obs = get_observation()
        action = agent.act(obs)
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
                reward += 1
        
        for pipe in pipes:
            pipe.draw(screen)

        rectangle.draw(screen)

        # Check for collisions
        done = False
        reward = 0.1
        for pipe in pipes:
            if pygame.Rect(rectangle.x, rectangle.y, RECT_WIDTH, RECT_HEIGHT).colliderect(pipe.top) or \
               pygame.Rect(rectangle.x, rectangle.y, RECT_WIDTH, RECT_HEIGHT).colliderect(pipe.bottom):
                game_active = False
                done = True
                reward = -1
                break
        if rectangle.y <= 0 or rectangle.y + RECT_HEIGHT >= SCREEN_HEIGHT:
            game_active = False
            done = True
            reward = -1

        next_obs = get_observation()
        
        agent.remember(obs, action, reward, next_obs, done)
        episode_reward += reward
        
        # Update the agent
        agent.replay(batch_size=batch_size)

        # Display the score
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        pygame.time.Clock().tick(30)

    time2 = time.time() - time1
    episode_rewards.append(episode_reward)
    avg = np.average(episode_rewards)
    
    print(f"Episode {episode+1}| {episode_reward} | improvment: {avg:.2f} | {score} | {time2:.2f}")
    with open(f"models/Training_{len(os.listdir('models/'))}/outputs.txt", "a") as file:
        file.write(f"{episode} | improvment: {avg} | {score} | {time2:.2f}\n")


# Save the trained model
number = len(os.listdir('models/'))+1
torch.save(agent.model.state_dict(), f"models/flappy{number}.pth")

# Plot the rewards
plt.figure(figsize=(12, 6))
plt.plot(range(len(episode_rewards)), episode_rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

pygame.quit()


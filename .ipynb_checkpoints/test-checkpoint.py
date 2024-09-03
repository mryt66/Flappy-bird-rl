import pygame
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from agent import DQNAgent

# Initialize pygame
pygame.init()

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

def take_action(action):
    if action == 1:  # Jump
        rectangle.jump()

# Initialize game components
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Rectangle")

rectangle = Rectangle()
pipes = []
pipe_timer = 0
score = 0
font = pygame.font.SysFont(None, 36)
game_active = True

model_path = f"models/flappy{len(os.listdir('models/'))}.pth"
agent = DQNAgent(state_dim=4, action_dim=2, lr=0.001, gamma=0.99, epsilon=0.01, epsilon_decay=0.995, buffer_size=10000)
agent.model.load_state_dict(torch.load(model_path))

episode_rewards = []

def reset_game():
    rectangle.x = SCREEN_WIDTH // 4
    rectangle.y = SCREEN_HEIGHT // 2
    rectangle.y_speed = 0
    
    pipes = []
    pipe_timer = 0
    score = 0
    game_active = True
    
    return rectangle, pipes, pipe_timer, score, game_active

def render_game(screen, font, rectangle, pipes, score):
    screen.fill(BLACK)
    
    for pipe in pipes:
        pygame.draw.rect(screen, WHITE, pipe.top)
        pygame.draw.rect(screen, WHITE, pipe.bottom)
    
    pygame.draw.rect(screen, RED, (rectangle.x, rectangle.y, RECT_WIDTH, RECT_HEIGHT))
    
    text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(text, (10, 10))
    
    pygame.display.flip()

def test_model():
    clock = pygame.time.Clock()
    running = True
    
    rectangle, pipes, pipe_timer, score, game_active = reset_game()
    
    while running:
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        obs = get_observation(rectangle, pipes)
        action = agent.act(obs)
        take_action(action)
        rectangle.apply_gravity()
        
        pipe_timer += 1
        if pipe_timer > 90:
            pipes.append(Pipe())
            pipe_timer = 0
        
        for pipe in pipes[:]:
            pipe.move()
            if pipe.off_screen():
                pipes.remove(pipe)
                score += 1
        
        done = False
        reward = 0.1
        for pipe in pipes:
            if pygame.Rect(rectangle.x, rectangle.y, RECT_WIDTH, RECT_HEIGHT).colliderect(pipe.top) or \
               pygame.Rect(rectangle.x, rectangle.y, RECT_WIDTH, RECT_HEIGHT).colliderect(pipe.bottom):
                game_active = False
                done = True
                break
        if rectangle.y <= 0 or rectangle.y + RECT_HEIGHT >= SCREEN_HEIGHT:
            game_active = False
            done = True
        
        render_game(screen, font, rectangle, pipes, score)
        
        if not game_active:
            running = False
    
    pygame.quit()

# Run the test
test_model()

# Visualize scores
episode_rewards.append(score)
plt.figure(figsize=(10, 6))
plt.plot(range(len(episode_rewards)), episode_rewards)
plt.title("Flappy Rectangle Scores")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()

import os
import pygame
import random
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from agent_cuda import DQNAgent
from parameters import (
    lr,
    gamma,
    epsilon,
    epsilon_decay,
    buffer_size,
    penalty,
    DEVICE,
    batch_size,
    target_update,
)

RENDER = False
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
JUMP_STRENGTH = -10
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
        self.top = pygame.Rect(
            SCREEN_WIDTH,
            0,
            PIPE_WIDTH,
            random.randint(15, SCREEN_HEIGHT - PIPE_GAP - 15),
        )
        self.bottom = pygame.Rect(
            SCREEN_WIDTH,
            self.top.height + PIPE_GAP,
            PIPE_WIDTH,
            SCREEN_HEIGHT - self.top.height - PIPE_GAP,
        )
        self.passed = False

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.top)
        pygame.draw.rect(screen, WHITE, self.bottom)

    def move(self):
        self.top.x -= PIPE_SPEED
        self.bottom.x -= PIPE_SPEED

    def off_screen(self):
        return self.top.right < 0

    def has_passed(self, rect_x):
        if not self.passed and self.top.right < rect_x:
            self.passed = True
            return True
        return False


def get_observation(rectangle, pipes):
    dist_horizontal, dist_vertically = rectangle.distance(pipes)
    if dist_horizontal is None or dist_vertically is None:
        dist_horizontal = SCREEN_WIDTH
        dist_vertically = SCREEN_HEIGHT // 2
    normalized_horizontal = dist_horizontal / SCREEN_WIDTH
    normalized_vertical = dist_vertically / SCREEN_HEIGHT
    rect_y_normalized = rectangle.y / SCREEN_HEIGHT
    rect_y_speed_normalized = rectangle.y_speed / JUMP_STRENGTH
    observation = np.array(
        [
            normalized_horizontal,
            normalized_vertical,
            rect_y_normalized,
            rect_y_speed_normalized,
        ]
    )
    return observation


def take_action(rectangle, action):
    if action == 1:
        rectangle.jump()


class Environment:
    def __init__(self, render=False):
        self.render = render
        self.screen = (
            pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            if self.render
            else None
        )
        self.clock = pygame.time.Clock() if self.render else None
        if self.render:
            pygame.display.set_caption("Flappy Rectangle")
            self.font = pygame.font.SysFont(None, 36)
        self.reset()

    def reset(self):
        self.rectangle = Rectangle()
        self.pipes = []
        self.pipe_timer = 0
        self.score = 0
        self.game_active = True
        return get_observation(self.rectangle, self.pipes)

    def step(self, action):
        reward = 0.1
        done = False
        take_action(self.rectangle, action)
        self.rectangle.apply_gravity()
        self.pipe_timer += 1
        if self.pipe_timer > 90:
            self.pipes.append(Pipe())
            self.pipe_timer = 0
        pipes_to_remove = []
        for pipe in self.pipes:
            pipe.move()
            if pipe.has_passed(self.rectangle.x):
                self.score += 1
                reward += 1
            if pipe.off_screen():
                pipes_to_remove.append(pipe)
        for pipe in pipes_to_remove:
            self.pipes.remove(pipe)
        if self.render:
            self.screen.fill(BLACK)
            for pipe in self.pipes:
                pipe.draw(self.screen)
            self.rectangle.draw(self.screen)
            score_text = self.font.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(score_text, (10, 10))
            pygame.display.flip()
            self.clock.tick(30)
        for pipe in self.pipes:
            rect = pygame.Rect(
                self.rectangle.x, self.rectangle.y, RECT_WIDTH, RECT_HEIGHT
            )
            if rect.colliderect(pipe.top) or rect.colliderect(pipe.bottom):
                done = True
                reward = penalty
                self.game_active = False
                break
        if self.rectangle.y <= 0 or self.rectangle.y + RECT_HEIGHT >= SCREEN_HEIGHT:
            done = True
            reward = penalty
            self.game_active = False
        next_observation = get_observation(self.rectangle, self.pipes)
        return next_observation, reward, done


def main():
    num_agents = 5
    shared_env = Environment(render=False)
    agents = [
        DQNAgent(
            state_dim=4,
            action_dim=2,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=0.01,
            buffer_size=buffer_size,
        )
        for _ in range(num_agents)
    ]
    episode_rewards = [[] for _ in range(num_agents)]
    episode_scores = [[] for _ in range(num_agents)]
    num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    os.makedirs("models", exist_ok=True)
    for episode in range(num_episodes):
        best_score = -float("inf")
        best_agent_index = -1
        for i in range(num_agents):
            state = shared_env.reset()
            done = False
            total_reward = 0
            total_score = 0
            while not done:
                action = agents[i].act(state)
                next_state, reward, done = shared_env.step(action)
                agents[i].remember(state, action, reward, next_state, done)
                agents[i].replay()
                state = next_state
                total_reward += reward
                total_score = shared_env.score
            agents[i].decay_epsilon()
            episode_rewards[i].append(total_reward)
            episode_scores[i].append(total_score)
            if total_score > best_score:
                best_score = total_score
                best_agent_index = i
        if (episode + 1) % target_update == 0:
            for agent in agents:
                agent.update_target_network()
        best_agent_reward = episode_rewards[best_agent_index][episode]
        best_agent_score = episode_scores[best_agent_index][episode]
        best_agent_epsilon = agents[best_agent_index].epsilon
        print(
            f"Episode {episode+1}, Reward: {best_agent_reward:.2f}, Score: {best_agent_score}, Epsilon: {best_agent_epsilon:.4f}"
        )
    for i, agent in enumerate(agents):
        torch.save(agent.policy_net.state_dict(), f"models/flappy_agent_{i+1}.pth")
    overall_best_score = -float("inf")
    overall_best_agent_index = -1
    for i in range(num_agents):
        avg_score = np.mean(episode_scores[i])
        if avg_score > overall_best_score:
            overall_best_score = avg_score
            overall_best_agent_index = i
    print(
        f"\nOverall Best Agent Index: {overall_best_agent_index + 1}, Average Score: {overall_best_score:.2f}"
    )
    pygame.quit()


if __name__ == "__main__":
    main()

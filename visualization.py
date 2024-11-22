# visualize_agent.py
import os
import torch
import pygame
import numpy as np
from agent_cuda import DQNAgent
from game import Environment, get_observation
from parameters import DEVICE

pygame.init()

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


def load_trained_model(model_path, state_dim, action_dim):
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0,
        gamma=0,
        epsilon=0,
        epsilon_decay=0,
        epsilon_min=0,
        buffer_size=0,
    )
    agent.policy_net.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    agent.policy_net.eval()
    return agent


def visualize_agent(agent, env):
    state = env.reset()
    done = False
    total_score = 0

    while not done:
        env.screen.fill(BLACK)
        for pipe in env.pipes:
            pipe.draw(env.screen)
        env.rectangle.draw(env.screen)
        score_text = env.font.render(f"Score: {env.score}", True, WHITE)
        env.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        env.clock.tick(60)

        action = agent.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_score = env.score

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


def main():
    model_path = "models/s32_e1208.pth"
    state_dim = 4
    action_dim = 2
    agent = load_trained_model(model_path, state_dim, action_dim)
    env = Environment(render=True)
    visualize_agent(agent, env)
    pygame.quit()


if __name__ == "__main__":
    main()
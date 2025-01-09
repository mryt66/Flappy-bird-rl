import os
import json
import numpy as np
import torch
import time
from agent_cuda import DQNAgent
from parameters import DEVICE
import warnings

warnings.simplefilter("ignore")

STATE_FILE = "C:/FlappyBirdBridge/state.json"
ACTION_FILE = "C:/FlappyBirdBridge/action.json"
ISDONE_FILE = "C:/FlappyBirdBridge/isdone.json"


def read_state():
    """Read the game state from a JSON file."""
    if not os.path.exists(STATE_FILE):
        return None, None, None
    with open(STATE_FILE, "r") as file:
        try:
            data = json.load(file)
            state = np.array(
                [
                    data["pipe_x"],  # Normalized pipe horizontal distance
                    data["pipe_y"],  # Normalized pipe vertical distance
                    data["rect_y"],  # Normalized rectangle position
                    data["rect_y_speed"],  # Normalized rectangle speed
                ]
            )
            reward = data["reward"]
            score = data["score"]
            return state, reward, score
        except Exception as e:
            # print(f"Error reading state: {e}")
            return None, None, None


def write_action(action):

    try:
        """Write the chosen action to a JSON file safely."""
        temp_file = ACTION_FILE + ".tmp"
        with open(temp_file, "w") as file:
            json.dump({"action": int(action)}, file)
        os.replace(temp_file, ACTION_FILE)
        # print(f"Action: {action}")
        os.remove(ISDONE_FILE)
    except Exception as e:
        # print(f"Error writing action: {e}")
        return None


def main():
    state_dim = 4
    action_dim = 2
    model_path = "models/s20_e1670.pth"
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

    agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.policy_net.eval()

    print("Python agent for visualization started. Waiting for Unity...")

    episode=4478
    epsilon_global=0.9985**4478
    print(f"Starting episode: {episode}, Epsilon: {epsilon_global:.4f}")
    
    best_agent_epsilon=0.1 #!!!!!!!!
    best_agent_reward=0
    best_agent_score=0
    total_reward=0
    while True:
        if not os.path.exists(ISDONE_FILE):
            # print("Czekam na plik isdone.json...")
            continue
        else:
            state, reward, score = read_state()
            if state is not None:
                # print(f"State: {state}")
                action = agent.act(state)
                write_action(action)
                # print(reward)
                total_reward += reward
               
        if reward==-10:
            episode+=1
            best_agent_reward = total_reward
            best_agent_score = score
            total_reward=0
            epsilon_global*=0.9985
            epsilon_global=0.01
            print()
            print(
                f"Episode {episode}, Reward: {best_agent_reward:.2f}, Score: {best_agent_score}, Epsilon: {epsilon_global:.4f}"
            )


if __name__ == "__main__":
    main()

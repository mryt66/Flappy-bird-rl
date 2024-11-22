import os
import json
import numpy as np
from agent_cuda import DQNAgent
from parameters import DEVICE, lr, gamma, epsilon, epsilon_decay, buffer_size

STATE_FILE = "C:/FlappyBirdBridge/state.json"
ACTION_FILE = "C:/FlappyBirdBridge/action.json"

def read_state():
    """Read the game state from a JSON file."""
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE, "r") as file:
        try:
            data = json.load(file)
            state = np.array([
                data["pipe_x"] / 400.0,  # Normalized pipe horizontal distance
                data["pipe_y"] / 600.0,  # Normalized pipe vertical distance
                data["rect_y"] / 600.0,  # Normalized rectangle position
                data["rect_y_speed"] / 10.0  # Normalized rectangle speed
            ])
            return state
        except Exception as e:
            print(f"Error reading state: {e}")
            return None

def write_action(action):
    """Write the chosen action to a JSON file."""
    with open(ACTION_FILE, "w") as file:
        json.dump({"action": int(action)}, file)

def main():
    state_dim = 4
    action_dim = 2
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        buffer_size=buffer_size,
    )
    
    model_path = "models/flappy_agent_2.pth"
    if os.path.exists(model_path):
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.policy_net.eval()
    
    print("Python agent started. Waiting for Unity...")
    
    while True:
        state = read_state()
        if state is not None:
            action = agent.act(state)
            write_action(action)

if __name__ == "__main__":
    main()

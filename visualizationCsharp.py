import json
import time
import torch
from agent_cuda import DQNAgent
from parameters import DEVICE

generalPath= "C:/FlappyBirdBridge/"

# Załaduj model agenta
agent = DQNAgent(
    state_dim=4,
    action_dim=2,
    lr=0,
    gamma=0,
    epsilon=0,
    epsilon_decay=0,
    epsilon_min=0,
    buffer_size=0,
)
agent.policy_net.load_state_dict(torch.load("models/flappy_agent_2.pth", map_location=DEVICE))
agent.policy_net.eval()

def read_game_state(file_path=generalPath + "state.json"):
    """Czytaj stan gry zapisany przez Unity."""
    with open(file_path, "r") as file:
        return json.load(file)

def write_action(action, file_path=generalPath + "action.json"):
    """Zapisz akcję do pliku JSON, aby Unity mogło ją odczytać."""
    with open(file_path, "w") as file:
        json.dump({"action": action}, file)

def main():
    print("Agent Python startuje...")
    while True:
        try:
            # Odczyt stanu gry
            game_state = read_game_state()

            # Pobranie obserwacji z Unity
            state = [
                game_state["rect_y"] / 600,          # Normalizacja Y
                game_state["rect_y_speed"] / 10,    # Normalizacja prędkości
                game_state["pipe_x"] / 400,         # Normalizacja pozycji X rury
                game_state["pipe_y"] / 600        # Normalizacja luki
            ]

            # Obliczenie akcji
            action = agent.act(state)

            # Zapis akcji
            write_action(action)

        except FileNotFoundError:
            print("Czekam na plik state.json...")
            time.sleep(0.1)
        except json.JSONDecodeError:
            print("Plik state.json jest uszkodzony. Ignoruję...")
            time.sleep(0.1)

        # Małe opóźnienie dla płynności
        time.sleep(0.05)

if __name__ == "__main__":
    main()

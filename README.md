# Flappy Bird AI
Author: Marcin Ryt, Konrad Wołoszyn




Unity
=============
The main part of the game is developed in Unity, providing the visual and interactive components of the Flappy Bird game. The Unity project includes the following:

Game Visualization:
A fully designed 2D environment with the bird, pipes, background, and physics-based interactions for realistic gameplay.

Training Integration:
The training process involves both Unity and Python. Unity serves as the environment simulator, while Python handles the AI agent's logic and learning.

Unity-Python Connection
The communication between Unity and Python is facilitated through observation files, enabling:

State Sharing: Unity sends the current game state to Python for processing.
Action Responses: Python computes the optimal action (flap or no-flap) and sends it back to Unity.

## How to run
Note: this version is just simple visualization in Pygame
To run the visualiztion of trained model, type in terminal:
```
   git clone https://github.com/mryt66/Flappy-bird-rl.git
   cd Flappy-bird-rl
```

The sprites version is not uploaded to repository
To run the visualization run this command:
```
   python visualization.py
```

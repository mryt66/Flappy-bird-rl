# Flappy Bird AI

<img src="gifs/Film.gif" alt="Flappy Bird Gameplay" width="200">


How does it work?
=============
The main part of the game is developed in Unity, providing the visual and interactive components of the Flappy Bird game.
Game includes a fully designed 2D environment with the bird, pipes, background, and physics-based interactions for realistic gameplay.
The training process involves both Unity and Python. Unity serves as the environment simulator, while Python handles the AI agent's logic and learning.

Unity-Python Connection
The communication between Unity and Python is facilitated through observation files, enabling:
State Sharing: Unity sends the current game state to Python for processing.
Action Responses: Python computes the optimal action (flap or no-flap) and sends it back to Unity.

## How to run
'Note: this version is just simple visualization in Pygame.'

The sprites version is not uploaded to repository
To run the basic visualization run this command:
```
   python visualization.py
```

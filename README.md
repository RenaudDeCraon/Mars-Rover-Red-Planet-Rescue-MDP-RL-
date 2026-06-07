# Mars Rover: Red Planet Rescue (MDP & Reinforcement Learning)

An AI project implementing Value Iteration and Q-Learning algorithms in a gridworld environment representing a Mars Rover navigation scenario.

## Features

- **Value Iteration**: Computes the optimal policy for a given Markov Decision Process (MDP) iteratively under uncertainty.
- **Q-Learning**: An model-free reinforcement learning algorithm that learns optimal state-action values ($Q$-values) through trial and error.
- **Gridworld Simulation**: Custom grid layouts (e.g., canyon_grid) representing terrain navigation obstacles, hazards, and rewards.
- **Interactive Mode**: Playable mode allowing keyboard control to navigate the rover manually.

## Project Structure

- `mars_rover.py`: The main entry point to execute gridworld configurations and training.
- `util.py`: Mathematical helper functions and data structures.
- `project_params.py`: Configuration details for grid rewards, discounts, and transitions.
- `autograder.py`: Autograder validation script to test Q-learning and Value Iteration results.

## Usage

### Run Keyboard Manual Control Mode
```bash
python mars_rover.py -m
```

### Run Value Iteration (100 Iterations, 10 Simulations)
```bash
python mars_rover.py -a value -i 100 -k 10
```

### Run Q-Learning (100 Simulations)
```bash
python mars_rover.py -a q -k 100
```

### Run Autograder Tests
```bash
python autograder.py
```

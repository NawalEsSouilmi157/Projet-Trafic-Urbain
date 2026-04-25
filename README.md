# Intelligent Urban Traffic Management (Q-Learning)

This project implements an intelligent traffic light controller using a manual implementation of the Q-Learning algorithm. It is developed as part of an AI university project for urban traffic management.

## Project Structure

- `intersection_sim.py`: Simulation environment modeling a 4-way intersection. It incorporates Poisson arrivals for vehicles, queues, and logic for transitioning traffic light phases.
- `agent.py`: A from-scratch implementation of the Q-Learning agent, managing the Q-table, applying the Bellman update equation, and executing the $\epsilon$-greedy policy with exponential decay.
- `baseline.py`: A simple fixed-duration traffic light policy that switches light phases periodically, provided for performance comparison.
- `train.py`: Script to train the Q-Learning agent on the simulation environment, generating a learned `q_table.pkl` and a convergence curve visualization `convergence.png`.
- `evaluate.py`: Script to evaluate and compare the Q-Learning agent against the baseline agent across two distinct scenarios: "Balanced Traffic" and "Asymmetric Traffic". It generates `evaluation_results.png`.

## MDP Formalization

As defined by the project requirements:
- **State**: The discretized number of vehicles in queues (0-5 per branch) plus the current traffic light phase (*0: N/S Green, 1: N/S Orange, 2: E/W Green, 3: E/W Orange*).
- **Actions**: `0` (Maintain Phase) and `1` (Switch Phase). When switching to a new green phase, the environment enforces a 1-step orange transition phase where no departures occur.
- **Reward**: The negative sum of the lengths of all queues, designed to minimize the total accumulated waiting time.
- **Hyperparameters**: 
  - Discount Factor ($\gamma$): 0.95
  - Learning Rate ($\alpha$): 0.1
  - Exploration Policy: $\epsilon$-greedy decaying from 1.0 to 0.01 with a decay factor of 0.995.

## Installation

Ensure you have Python 3 installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Project

1. **Train the agent**:
   ```bash
   python train.py
   ```
   This will train the Q-Learning agent over 1500 episodes, saving its learned Q-table (`q_table.pkl`), and generate a convergence plot (`convergence.png`).

2. **Evaluate the agent**:
   ```bash
   python evaluate.py
   ```
   *Make sure you run `train.py` first.* This evaluation script runs a simulation comparing the trained Q-Learning agent against the fixed baseline under balanced and asymmetric traffic conditions, saving visual comparisons to `evaluation_results.png`.

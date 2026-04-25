import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from intersection_sim import IntersectionEnv
from agent import QLearningAgent
from baseline import FixedDurationAgent

def evaluate_agent(env, agent_type='qlearning', q_table=None, steps=500):
    if agent_type == 'qlearning':
        # Epsilon 0.0 for pure exploitation during evaluation
        agent = QLearningAgent(epsilon=0.0) 
        if q_table:
            agent.q_table = q_table
    else:
        # Fixed baseline agent that switches every 15 steps
        agent = FixedDurationAgent(duration=15)
        
    state = env.reset()
    total_waiting_time = 0
    queues_history = []
    
    for _ in range(steps):
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        
        # Reward is negative, we accumulate it as wait time
        total_waiting_time += -reward
        queues_history.append(np.sum(env.queues))
        
    return total_waiting_time, queues_history

def run_evaluation():
    # Load trained Q-table
    model_path = os.path.join(os.path.dirname(__file__), 'q_table.pkl')
    try:
        with open(model_path, 'rb') as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {model_path} not found.")
        print("Please run train.py first to generate the trained Q-table.")
        return

    # Two scenarios requested for evaluation comparison
    scenarios = {
        "Balanced Traffic": (0.4, 0.4, 0.4, 0.4),         # Symmetrical load
        "Asymmetric Traffic": (0.8, 0.8, 0.1, 0.1)        # Heavy N/S traffic, light E/W traffic
    }
    
    results = {}
    
    for name, lambdas in scenarios.items():
        print(f"\n--- Testing Scenario: {name} ---")
        print(f"Arrival Rates (Lambda): N/S={lambdas[0]}, E/W={lambdas[2]}")
        env = IntersectionEnv(arrivals_lambda=lambdas)
        
        wait_q, hist_q = evaluate_agent(env, 'qlearning', q_table)
        wait_baseline, hist_b = evaluate_agent(env, 'baseline')
        
        print(f"Q-Learning   - Total Waiting Vehicles: {wait_q:.0f}")
        print(f"Baseline     - Total Waiting Vehicles: {wait_baseline:.0f}")
        
        results[name] = {
            'Q-Learning': hist_q,
            'Baseline': hist_b
        }

    # Plot comparisons
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (name, data) in enumerate(results.items()):
        axes[i].plot(data['Q-Learning'], label='Q-Learning Agent', alpha=0.8, color='blue')
        axes[i].plot(data['Baseline'], label='Fixed Baseline', alpha=0.8, color='orange')
        axes[i].set_title(name, fontsize=14)
        axes[i].set_xlabel('Time Steps', fontsize=12)
        axes[i].set_ylabel('Total Number of Vehicles Waiting', fontsize=12)
        axes[i].legend()
        axes[i].grid(True, alpha=0.5)
        
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'evaluation_results.png')
    plt.savefig(plot_path)
    print(f"\nSaved evaluation comparison plots to {plot_path}")

if __name__ == "__main__":
    run_evaluation()

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from intersection_sim import IntersectionEnv
from agent import QLearningAgent

def train(episodes=5000, steps_per_episode=200):
    print("Starting Q-Learning Training...")
    # Initialize environment with balanced traffic arrivals
    env = IntersectionEnv(arrivals_lambda=(0.4, 0.4, 0.4, 0.4))
    
    # Initialize Q-Learning agent with tuned hyperparameters
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.9995)
    
    rewards_history = []
    
    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(steps_per_episode):
            # Agent selects action
            action = agent.choose_action(state)
            
            # Environment processes action and returns new state and reward
            next_state, reward, done, _ = env.step(action)
            
            # Agent learns from the transition
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            episode_reward += reward
            
        # Decay epsilon eagerly early on
        agent.update_epsilon()
        rewards_history.append(episode_reward)
        
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}/{episodes} - Reward: {episode_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
            if (ep + 1) % 1000 == 0:
                unique_states = len(set([k[0] for k in agent.q_table.keys()]))
                print(f">>> Discovered Unique States: {unique_states}")
            
    # Calculate a moving average for smoothing the curve
    window = 50
    smoothed_rewards = [np.mean(rewards_history[max(0, i-window):i+1]) for i in range(len(rewards_history))]
    
    # Plotting the convergence curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, alpha=0.3, color='blue', label='Raw Reward')
    plt.plot(smoothed_rewards, color='red', linewidth=2, label='Smoothed Reward (EMA)')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward (Negative Waiting Time)')
    plt.title('Q-Learning Training Convergence')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(__file__), 'convergence.png')
    plt.savefig(plot_path)
    print(f"\nSaved convergence plot to {plot_path}")
    
    # Save the learned Q-Table
    model_path = os.path.join(os.path.dirname(__file__), 'q_table.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(agent.q_table, f)
    print(f"Saved Q-Table to {model_path}")
        
    return agent

if __name__ == "__main__":
    train()

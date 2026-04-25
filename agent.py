import numpy as np
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, num_actions=2):
        self.alpha = alpha       # Learning rate
        self.gamma = gamma       # Discount factor
        self.epsilon = epsilon   # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        
        # Q-table: dictionary mapping (state, action) -> value
        # Using a dict to dynamically allocate state spaces as encountered
        self.q_table = {}

    def get_q(self, state, action):
        """Retrieve Q value from table. Initialize to 0.0 if unseen."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, valid_actions=None):
        """Epsilon-greedy policy for action selection."""
        if valid_actions is None:
            valid_actions = list(range(self.num_actions))
            
        # Explore
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploit
        q_values = [self.get_q(state, a) for a in valid_actions]
        max_q = max(q_values)
        
        # Break ties randomly to avoid always favoring the same action when Q-values are equal
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        """Update Q-table using the Bellman equation."""
        current_q = self.get_q(state, action)
        
        # Maximum expected future reward given the new state
        next_max_q = max([self.get_q(next_state, a) for a in range(self.num_actions)])
        
        # Q-Learning update rule
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q

    def update_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

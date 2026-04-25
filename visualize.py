import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
from intersection_sim import IntersectionEnv
from agent import QLearningAgent

def run_visualization(steps=200, interval=1000):
    """
    Runs a real-time visualization of the intersection using matplotlib.
    interval: Time in milliseconds between steps (slow-motion mode).
    """
    # 1. Load the trained Q-Table
    model_path = os.path.join(os.path.dirname(__file__), 'q_table.pkl')
    if not os.path.exists(model_path):
        print("Error: q_table.pkl not found. Please train the model first by running 'py train.py'.")
        return
        
    with open(model_path, 'rb') as f:
        q_table = pickle.load(f)
        
    # 2. Setup Environment and Agent using Balanced Traffic settings
    env = IntersectionEnv(arrivals_lambda=(0.4, 0.4, 0.4, 0.4))
    agent = QLearningAgent(epsilon=0.0) # Greedy deterministic policy (Exploitation)
    agent.q_table = q_table
    
    current_state = [env.reset()]
    current_step = [0]
    
    # 3. Setup Matplotlib Figure & Aesthetics
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#2b2b2b') # Dark mode background
    ax.set_facecolor('#2b2b2b')
    ax.set_title("Urban Traffic Q-Learning Control", color='white', fontsize=18, pad=20)
    
    # Draw roads (cross format) background layer
    ax.axhline(0, color='#404040', linewidth=60, zorder=0) # E/W Road
    ax.axvline(0, color='#404040', linewidth=60, zorder=0) # N/S Road
    
    # Hidden axes for neat display
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.axis('off')
    
    # UI Texts
    step_text = ax.text(-3.8, 3.5, '',     fontsize=14, color='white', fontweight='bold')
    action_text = ax.text(-3.8, 3.1, '',   fontsize=12, color='#00ffcc')
    reward_text = ax.text(-3.8, 2.7, '',   fontsize=12, color='#ff6b6b')
    reason_text = ax.text(-3.8, 2.3, '',   fontsize=11, color='#ffcc00')
    
    # Queue length text labels 
    qn_text = ax.text(0.4, 2.0, '',  ha='left', va='center', fontsize=12, color='white', fontweight='bold')
    qs_text = ax.text(0.4, -2.0, '', ha='left', va='center', fontsize=12, color='white', fontweight='bold')
    qe_text = ax.text(2.0, 0.4, '',  ha='center', va='bottom', fontsize=12, color='white', fontweight='bold')
    qw_text = ax.text(-2.0, 0.4, '', ha='center', va='bottom', fontsize=12, color='white', fontweight='bold')
    
    # Traffic Lights setup
    light_n = plt.Circle((0, 1.2), 0.3, color='red', zorder=3)
    light_s = plt.Circle((0, -1.2), 0.3, color='red', zorder=3)
    light_e = plt.Circle((1.2, 0), 0.3, color='red', zorder=3)
    light_w = plt.Circle((-1.2, 0), 0.3, color='red', zorder=3)
    
    for light in [light_n, light_s, light_e, light_w]:
        ax.add_patch(light)
        # Adds an inner glow
        ax.add_patch(plt.Circle(light.center, 0.15, color='white', alpha=0.3, zorder=4))
    
    # Vehicle Queues bars setup (dynamic rectangular bars to represent density)
    bar_width = 0.6
    bar_n = plt.Rectangle((-bar_width/2, 1.5), bar_width, 0, color='#4da6ff', alpha=0.9, zorder=2)
    bar_s = plt.Rectangle((-bar_width/2, -1.5), bar_width, 0, color='#4da6ff', alpha=0.9, zorder=2) 
    bar_e = plt.Rectangle((1.5, -bar_width/2), 0, bar_width, color='#4da6ff', alpha=0.9, zorder=2)
    bar_w = plt.Rectangle((-1.5, -bar_width/2), 0, bar_width, color='#4da6ff', alpha=0.9, zorder=2) 

    for bar in [bar_n, bar_s, bar_e, bar_w]:
        ax.add_patch(bar)
        
    last_action = [0]
    total_reward = [0]

    def update(frame):
        # Proceed logic for Environment & Agent step-by-step
        st = current_state[0]
        
        # Agent decides only in Green Phases. Env enforces the Orange.
        action = agent.choose_action(st)
        last_action[0] = action
        
        next_state, reward, done, info = env.step(action)
        current_state[0] = next_state
        current_step[0] += 1
        total_reward[0] += reward
        
        p = env.phase
        
        # Determine Color Codes 
        c_ns = 'red'
        c_ew = 'red'
        
        if p == 0:
            c_ns = '#00ff00' # Bright green
        elif p == 1:
            c_ns = '#ffa500' # Orange
        elif p == 2:
            c_ew = '#00ff00'
        elif p == 3:
            c_ew = '#ffa500'
            
        light_n.set_color(c_ns)
        light_s.set_color(c_ns)
        light_e.set_color(c_ew)
        light_w.set_color(c_ew)
        
        # Extract vehicle queue states
        qN, qS, qE, qW = env.queues
        
        qn_text.set_text(f"N: {qN} cars")
        qs_text.set_text(f"S: {qS} cars")
        qe_text.set_text(f"E: {qE} cars")
        qw_text.set_text(f"W: {qW} cars")
        
        # Scaling visuals - assumes max visualized queue of around 15 visually
        # Actual size grows dynamically
        scale = 2.0 / 15.0 
        
        bar_n.set_height(min(qN * scale, 2.0))
        
        # Adjust Y so south bar grows downward
        h_s = min(qS * scale, 2.0)
        bar_s.set_height(h_s)
        bar_s.set_y(-1.5 - h_s)
        
        bar_e.set_width(min(qE * scale, 2.0))
        
        # Adjust X so west bar grows leftward
        w_w = min(qW * scale, 2.0)
        bar_w.set_width(w_w)
        bar_w.set_x(-1.5 - w_w)
        
        # UI overlays
        step_text.set_text(f"Time Step: {current_step[0]}")
        reward_text.set_text(f"Avg Reward/step: {(total_reward[0] / current_step[0]):.2f}")
        
        action_str = "Switching Phase" if action == 1 else "Maintaining Green"
        if p in [1, 3]:
            action_str = "Env: Transition Phase (Delay)"
        
        action_text.set_text(f"Agent Action: {action_str}")
        
        reason_str = info.get('reason', '')
        reason_text.set_text(f"{reason_str}")
        
        return light_n, light_s, light_e, light_w, qn_text, qs_text, qe_text, qw_text, step_text, action_text, reward_text, reason_text, bar_n, bar_s, bar_e, bar_w
        
    # Set up animation
    # interval = milliseconds of pause between frames (Slow-Motion)
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=False, repeat=False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting visualizer in Slow-Motion... Close the window to stop.")
    # interval=1000 sets a 1-second pause per step allowing human observation of the AI decisions
    run_visualization(steps=300, interval=1000)

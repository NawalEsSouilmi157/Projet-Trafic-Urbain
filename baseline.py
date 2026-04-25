class FixedDurationAgent:
    def __init__(self, duration=10):
        self.duration = duration
        self.timer = 0
        
    def choose_action(self, state):
        """
        Policy based on a fixed timer duration for alternating traffic flow.
        state[-1] contains the current light phase:
        0: N/S Green, 1: N/S Orange, 2: E/W Green, 3: E/W Orange
        """
        phase = state[-1]
        
        # During orange transition phases, our action doesn't influence the change
        # but we maintain internal timer state to zero out after transition
        if phase in [1, 3]: 
            self.timer = 0
            return 0 # Maintain
            
        self.timer += 1
        
        # If the timer has reached the fixed duration for this green phase, trigger a switch
        if self.timer >= self.duration:
            self.timer = 0
            return 1 # Switch
        
        # Keep current phase otherwise
        return 0 # Maintain

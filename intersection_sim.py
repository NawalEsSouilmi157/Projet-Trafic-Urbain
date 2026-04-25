import numpy as np

class IntersectionEnv:
    def __init__(self, arrivals_lambda=(2.0, 2.0, 2.0, 2.0), max_queue=5):
        # arrivals_lambda: (lambda_N, lambda_S, lambda_E, lambda_W)
        self.arrivals_lambda = arrivals_lambda
        self.max_queue = max_queue
        
        # internal state: [queue_N, queue_S, queue_E, queue_W]
        self.queues = np.zeros(4, dtype=int)
        
        # phase: 0 = N/S Green, 1 = N/S Orange, 2 = E/W Green, 3 = E/W Orange
        self.phase = 0
        self.time_in_phase = 0
        
    def reset(self):
        self.queues = np.zeros(4, dtype=int)
        self.phase = 0
        self.time_in_phase = 0
        return self._get_state()

    def _get_state(self):
        # Discretize states up to max_queue (0-5 per branch) as required in the MDP formalization
        discretized_queues = np.clip(self.queues, 0, self.max_queue)
        # return tuple representing state (q_N, q_S, q_E, q_W, phase)
        return tuple(discretized_queues.tolist() + [self.phase])

    def step(self, action):
        """
        action: 0 (Maintain Phase), 1 (Change Phase)
        """
        # Poisson arrivals for each branch per step
        arrivals = np.random.poisson(self.arrivals_lambda)
        
        reason = ""
        
        # Override action logic based on safety rules during green phase
        if self.phase in [0, 2]:
            ns_queue = self.queues[0] + self.queues[1]
            ew_queue = self.queues[2] + self.queues[3]
            
            # 1. Switch if the other direction queue is much larger
            if self.phase == 0 and ew_queue > ns_queue + 10:
                action = 1
                reason = "Forced: E/W Queue is much larger"
            elif self.phase == 2 and ns_queue > ew_queue + 10:
                action = 1
                reason = "Forced: N/S Queue is much larger"
                
            # 2. Maximum green time (50 steps) to force switching
            if self.time_in_phase >= 50 and action == 0:
                action = 1
                reason = "Forced: Max Green Time (50 steps)"

        # Prevent erratic switching: Enforce a minimum green time of 3 steps
        if action == 1 and self.phase in [0, 2] and self.time_in_phase < 3:
            action = 0
            reason = "" # Min time takes precedence

        # Enable a capacity large enough to prevent the queues from exploding infinitely under traffic
        max_dep = 5 
        
        # If we are in an orange phase, the environment automatically transitions to the next green phase
        # regardless of the action chosen by the agent.
        if self.phase == 1: # N/S Orange
            self.phase = 2 # Switch to E/W Green
            self.time_in_phase = 0
            # Departures: N/S vehicles are stopped, E/W vehicles start moving
            departures = np.array([0, 0, min(self.queues[2]+arrivals[2], max_dep), min(self.queues[3]+arrivals[3], max_dep)])
        elif self.phase == 3: # E/W Orange
            self.phase = 0 # Switch to N/S Green
            self.time_in_phase = 0
            # Departures: N/S vehicles start moving, E/W vehicles are stopped
            departures = np.array([min(self.queues[0]+arrivals[0], max_dep), min(self.queues[1]+arrivals[1], max_dep), 0, 0])
        else:
            # We are in a green phase
            if action == 1: # Change phase requested
                if self.phase == 0:
                    self.phase = 1 # N/S Green -> N/S Orange
                elif self.phase == 2:
                    self.phase = 3 # E/W Green -> E/W Orange
                self.time_in_phase = 0
                # During the first step of orange transition, no vehicles depart
                departures = np.zeros(4, dtype=int)
            else: # Maintain phase
                if self.phase == 0:
                    # N/S Green: Allow departures on N and S
                    departures = np.array([min(self.queues[0]+arrivals[0], max_dep), min(self.queues[1]+arrivals[1], max_dep), 0, 0])
                elif self.phase == 2:
                    # E/W Green: Allow departures on E and W
                    departures = np.array([0, 0, min(self.queues[2]+arrivals[2], max_dep), min(self.queues[3]+arrivals[3], max_dep)])
                self.time_in_phase += 1

        # Update real queues and clip to a physical max (e.g. 50) to avoid runtime runaway errors 
        self.queues += arrivals
        self.queues -= departures
        self.queues = np.clip(self.queues, 0, 50)
        
        # Calculate Reward based on REAL queues. 
        # This gives the agent a gradient to clear queues even when they exceed the 
        # State Representation threshold limit (5), resolving the "stuck" blind-spot.
        reward = -np.sum(self.queues) / (4.0 * self.max_queue)
        
        # Explicit penalty for switching to discourage extremely high frequency phase changes
        if action == 1 and self.phase in [1, 3] and self.time_in_phase == 0:
            reward -= 0.5
            
        info = {'reason': reason}
        return self._get_state(), reward, False, info

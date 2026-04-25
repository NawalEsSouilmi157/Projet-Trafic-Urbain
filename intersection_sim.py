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
        
        # Calculate Reward strictly on formalization
        discretized_queues = np.clip(self.queues, 0, self.max_queue)
        reward = -float(np.sum(discretized_queues))
        
        reason = ""
        info = {'reason': reason}
        return self._get_state(), reward, False, info

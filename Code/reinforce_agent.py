import torch
import torch.nn as nn
import torch.nn.functional as F 
from constants import *
import numpy as np
from random import uniform, randint


class ReinforceNet(nn.Module):
    def __init__(self):
        super(ReinforceNet, self).__init__()
        self.fc1 = nn.Linear(198, 80)
        self.fc2 = nn.Linear(80, 6)
        
        self.softmax = nn.Softmax(dim=1)
        self.eligibility_traces = [torch.zeros_like(p) for p in self.parameters()]
        
        self.alpha = 0.01 #1e-2 #1e-4
        self.lam = 0.7 # 0
        self.gamma = 1
        
        self.register_buffer('rewards', torch.tensor([1, -1, 2, -2, 3, -3]))
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return self.softmax(x)
    
    def reset_eligbility_traces(self):
        self.eligibility_traces = [torch.zeros_like(p) for p in self.parameters()]
        
    def expected_value(self, probs):
        # rewards = torch.tensor([
        #     1, -1, 2, -2, 2, -3 # Win by backgammon set to 2 to be more risk averse
        # ])
        return torch.sum(probs*self.rewards, dim=1)
    
    def update_weights(self, current_state, next_state, reward, done):
        self.zero_grad()
        
        # 1. Convert states to tensors
        current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0)  # Shape: [1, 198]
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0) if next_state is not None else None
        
        # 2. Forward pass for current state
        probs_current = self(current_state_tensor)
        V_current = torch.sum(probs_current * self.rewards)  # Scalar expected value
        
        # 3. Backpropagate to compute gradients of V_current
        V_current.backward()
        
        # 4. Compute V_next (detached from computation graph)
        with torch.no_grad():
            if done or next_state is None:
                V_next = 0.0
            else:
                probs_next = self(next_state_tensor)
                V_next = torch.sum(probs_next * self.rewards).item()
        
        # 5. Calculate TD error
        delta = reward + self.gamma * V_next - V_current.item()
        
        # 6. Update eligibility traces and parameters
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                # Update eligibility traces: γλ * e_prev + ∇V_current
                if self.lam != 0:
                    self.eligibility_traces[i] = (
                        self.gamma * self.lam * self.eligibility_traces[i] + param.grad
                    )
                else:
                    self.eligibility_traces[i] = param.grad
                
                # Update weights: α * δ * e
                param.data += self.alpha * delta * self.eligibility_traces[i]
        
        return delta
    
    def select_action(self, encoded_boards):
        board_tensors = torch.FloatTensor(np.array(encoded_boards))
            # outcome_probs = []
        with torch.no_grad():
            outcome_probs = self(board_tensors)  # Shape: [num_boards, 6]
            expected_values = self.expected_value(outcome_probs)  # Shape: [num_boards]

        action_idx = torch.argmax(expected_values).item()  # Pick board with highest expected value
        return action_idx





class ReinforceNet3(nn.Module):
    def __init__(self):
        super(ReinforceNet3, self).__init__()
        self.fc1 = nn.Linear(291, 160)
        self.fc2 = nn.Linear(160, 6)
        
        self.softmax = nn.Softmax(dim=1)
        self.eligibility_traces = [torch.zeros_like(p) for p in self.parameters()]
        
        self.alpha = 0.01 #1e-2 #1e-4
        self.lam = 0.7 # 0
        self.gamma = 1
        self.epsilon = 0.1
        self.register_buffer('rewards', torch.tensor([1, -1, 2, -2, 3, -3]))
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return self.softmax(x)
    
    def reset_eligbility_traces(self):
        self.eligibility_traces = [torch.zeros_like(p) for p in self.parameters()]
        
    def expected_value(self, probs):
        # rewards = torch.tensor([
        #     1, -1, 2, -2, 2, -3 # Win by backgammon set to 2 to be more risk averse
        # ])
        return torch.sum(probs*self.rewards, dim=1)
    
    def update_weights(self, current_state, next_state, reward, done):
        self.zero_grad()
        
        # 1. Convert states to tensors
        current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0)  # Shape: [1, 198]
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0) if next_state is not None else None
        
        # 2. Forward pass for current state
        probs_current = self(current_state_tensor)
        V_current = torch.sum(probs_current * self.rewards)  # Scalar expected value
        
        # 3. Backpropagate to compute gradients of V_current
        V_current.backward()
        
        # 4. Compute V_next (detached from computation graph)
        with torch.no_grad():
            if done or next_state is None:
                V_next = 0.0
            else:
                probs_next = self(next_state_tensor)
                V_next = torch.sum(probs_next * self.rewards).item()
        
        # 5. Calculate TD error
        delta = reward + self.gamma * V_next - V_current.item()
        
        # 6. Update eligibility traces and parameters
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                # Update eligibility traces: γλ * e_prev + ∇V_current
                if self.lam != 0:
                    self.eligibility_traces[i] = (
                        self.gamma * self.lam * self.eligibility_traces[i] + param.grad
                    )
                else:
                    self.eligibility_traces[i] = param.grad
                
                # Update weights: α * δ * e
                param.data += self.alpha * delta * self.eligibility_traces[i]
        
        return delta
    
    def select_action(self, encoded_boards):
        # Epsilon-greedy exploration
        # print(f"Select action between {len(encoded_boards)} boards")
        if self.epsilon > uniform(0,1):
            return randint(0, len(encoded_boards)-1)
        board_tensors = torch.FloatTensor(np.array(encoded_boards))
            # outcome_probs = []
        with torch.no_grad():
            outcome_probs = self(board_tensors)  # Shape: [num_boards, 6]
            expected_values = self.expected_value(outcome_probs)  # Shape: [num_boards]

        action_idx = torch.argmax(expected_values).item()  # Pick board with highest expected value
        return action_idx
    
   

"""
Gradient Descent Alg:
For each game:
    eligibility traces = zero vector
    s = initial board state
    For each timestep:
        a = action from policy
        Take action, observe reward (r), and next state s'
        delta = r + estimated value of s' - estimate value of s
            estimated value are 6-vectors for loss/win by bg, g or normal
        eligibility traces = LAMBDA*eligibility traces + estimated value of s_t .backwards
        weights = weights + ALPHA*delta*eligibiliity traces
        s = s'


"""
import torch
import torch.nn as nn
import torch.nn.functional as F 

class ReinforceNet(nn.Module):
    def __init__(self):
        super(ReinforceNet, self).__init__()
        self.fc1 = nn.Linear(198, 80)
        self.fc2 = nn.Linear(80, 6)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x
        
    def update_weights(self, p, p_next):
        self.zero_grad()
        
        p.backward()
        
        with torch.no_grad():
            td_error = p_next - p
            
        
model = ReinforceNet()
total_params = sum(p.numel() for p in model.parameters())
print(total_params)


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
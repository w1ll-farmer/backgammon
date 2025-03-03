import torch
from torch import nn
# 1️⃣ Define the Same Model Structure
class BackgammonNet(nn.Module):
    def __init__(self):
        super(BackgammonNet, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(0.3),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# 3️⃣ Make Predictions
def predict(board_state, epoch):
    model = BackgammonNet()
    suffix = "" if epoch is None else f"_{epoch}"
    prefix = "Code/" if epoch is None else ""
    model.load_state_dict(torch.load(f"{prefix}backgammon_model{suffix}.pth"))
    model.eval()  # Set to evaluation mode (disables dropout, etc.)

    board_tensor = torch.tensor(board_state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        equity = model(board_tensor).item()  # Get the single output
    return equity


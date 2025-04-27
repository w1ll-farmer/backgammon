import torch
from torch import nn
import torch.nn.functional as F
from turn import *


# 1️⃣ Define the Same Model Structure
class BGNet(nn.Module):
    def __init__(self):
        super(BGNet, self).__init__()
        # Shared input-to-hidden layer: both board states use the same weights.
        self.fc_shared = nn.Linear(289, 12)
        # Hidden-to-output layer: we use a single layer that will be applied
        # to both hidden representations. In the forward pass we explicitly
        # flip the sign for one branch to enforce that its effect is the negative.
        self.fc_out = nn.Linear(12, 1)

    def forward(self, board_left, board_right):
        # Process left board: apply shared fc layer and ReLU activation.
        h_left = F.relu(self.fc_shared(board_left))
        # Process right board: same shared fc layer.
        h_right = F.relu(self.fc_shared(board_right))
        
        # Compute the output for each branch using the same fc_out layer.
        out_left = self.fc_out(h_left)
        # Explicitly invert the output from the right branch.
        out_right = -self.fc_out(h_right)
        
        # The network's decision is the sum of these contributions.
        # This is equivalent to computing the difference between the two evaluations.
        diff = out_left + out_right
        
        # Apply a sigmoid activation so that final output is between 0 and 1.
        final_output = torch.sigmoid(diff)
        return final_output

# 3️⃣ Make Predictions
def predict(left_board, right_board, input_vector_left, input_vector_right, epoch=None):
    model = BGNet()
    suffix = "" if epoch is None else f"{epoch}"
    prefix = "Code" #if epoch is None or epoch == 499 else ""
    if epoch is None:
        if all_past(left_board) and all_past(right_board):
            if all_checkers_home(1, left_board) and all_checkers_home(1, right_board):
                network = "bearoff_race"
            else:
                network = "midboard_race"
            model.load_state_dict(torch.load(os.path.join(f"{prefix}",f"{network}{suffix}.pth")))
        else:
            network = "contact"
            model.load_state_dict(torch.load(os.path.join("Code","backgammon_modelv1.pth")))
        
    elif epoch == "_50":
        if all_past(left_board) and all_past(right_board):
            if all_checkers_home(1, left_board) and all_checkers_home(1, right_board):
                network = "bearoff_race"
            else:
                network = "midboard_race"
            model.load_state_dict(torch.load(os.path.join(f"{prefix}",f"{network}{suffix}.pth")))
        else:
            # network = "contact" 
            model.load_state_dict(torch.load(os.path.join("Code","backgammon_modelv1.pth")))
         
    else:
        model.load_state_dict(torch.load(os.path.join("Code","backgammon_modelv1.pth")))
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    # input_vector_left = torch.tensor(convert_board(left_board), dtype=torch.float32).unsqueeze(0)
    # input_vector_right = torch.tensor(convert_board(right_board), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        equity = model(input_vector_left, input_vector_right).item()  # Get the single output
    return equity
    
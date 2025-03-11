import torch
from torch import nn
import torch.nn.functional as F
from turn import *
def convert_board(board):
    input_vector = []
    for i in range(24):
        point_encoding = convert_point(board[i])
        input_vector += point_encoding
    for i in range(24):
        input_vector.append(prob_opponent_can_hit(1, board, i))
    for i in range(24, 26):
        input_vector += convert_bar(board[i])
    _, home = get_home_info(1, board)
    _, opp_home = get_home_info(-1, board)
    # % home points occupied
    input_vector.append(len([i for i in home if i > 0])/6)
    # % opp home points occupied
    input_vector.append(len([i for i in opp_home if i > 0])/6)
    # % pieces in home
    input_vector.append(sum([i for i in home if i >0])/15)
    # Prime?
    input_vector.append(1 if calc_prime(board, 1) > 3 else 0)
    # pip count
    input_vector += decimal_to_binary(calc_pips(board, 1))
    input_vector += decimal_to_binary(calc_pips(board, -1))
    
    # chance blockade can't be passed
    input_vector.append(calc_blockade_pass_chance(board, 1))
    return input_vector

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
def predict(left_board, right_board, epoch=None):
    model = BGNet()
    suffix = "" if epoch is None else f"_{epoch}"
    prefix = "Code/" if epoch is None or epoch == 499 else ""
    model.load_state_dict(torch.load(f"{prefix}backgammon_model{suffix}.pth"))
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    input_vector_left = torch.tensor(convert_board(left_board), dtype=torch.float32).unsqueeze(0)
    input_vector_right = torch.tensor(convert_board(right_board), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        equity = model(input_vector_left, input_vector_right).item()  # Get the single output
    return equity


import torch
from torch import nn
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
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(289, 1)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(12, 12)
        # self.fc3 = nn.Linear(12, 1)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        return x

# 3️⃣ Make Predictions
def predict(board_state, epoch):
    model = SimpleModel()
    suffix = "" if epoch is None else f"_{epoch}"
    prefix = "Code/" if epoch is None or epoch == 499 else ""
    model.load_state_dict(torch.load(f"{prefix}backgammon_model{suffix}.pth"))
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    input_vector = convert_board(board_state)
    board_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        equity = model(board_tensor).item()  # Get the single output
    return equity


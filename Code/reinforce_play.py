import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import os

from constants import *
from testfile import invert_board
# from env import BackgammonEnv as env
from reinforce_agent import ReinforceNet

def encode_state(board_state, player):
        vector = []
        for point in range(24):
            vector += encode_point(board_state[point])
        vector.append(abs(board_state[24]/2))
        vector.append(board_state[25]/2)
        vector.append(abs(board_state[26]/15))
        vector.append(board_state[27]/15)
        vector.append(int(player == 1))
        vector.append(int(player == -1))
        return torch.FloatTensor(vector)

def encode_point(point):
        base = [0]* 8
        if point < 0:
            for i in range(0,3):
                if point < -i:
                    base[i] = 1
            if point < -3:
                base[3] = min((abs(point)-3)/2,1)
        elif point > 0:
            for i in range(4, 7):
                if point > i -4:
                    base[i] = 1
            if point > 3:
                base[7] = min((point-3)/2,1)
        return base
      
def reinforce_play(boards, moves, player):
    if player == -1:
        inverted_boards = [invert_board(i) for i in boards]
        encoded_boards = [encode_state(board, player) for board in inverted_boards]
    else:
        encoded_boards = [encode_state(board, player) for board in boards]
    model = ReinforceNet()
    model.load_state_dict(torch.load(os.path.join("Code","RL","reinforcement_200.pth"))['model_state_dict'])
    # encoded_boards = [encode_state(board) for board in inverted_boards]
    board_tensors = torch.FloatTensor(np.array(encoded_boards))
    with torch.no_grad():
        outcome_probs = model(board_tensors)  # Shape: [num_boards, 6]
        expected_values = model.expected_value(outcome_probs)  # Shape: [num_boards]

    action_idx = torch.argmax(expected_values).item()  # Pick board with highest expected value
    chosen_board = boards[action_idx]
    chosen_move = moves[action_idx]
    return chosen_move, chosen_board

def load_model(model, path="backgammon_model.pth"):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eligibility_traces = checkpoint['eligibility_traces']
    print(f"Model loaded from {path} (episode {checkpoint['episode']})")
    
    
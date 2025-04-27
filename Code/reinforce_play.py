import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import os

from constants import *
from testfile import invert_board
from turn import *
# from env import BackgammonEnv as env
from reinforce_agent import ReinforceNet, ReinforceNet3

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

def reinforce_lookahead(b, player, model, version):
    """Performs expectimax lookahead using model as evaluator

    Args:
        board (list(int)): Raw board encoding
        player (int): The adversary
        model (ReinforceV3): The model used to estimate the value of the position

    Returns:
        _type_: _description_
    """
    equities = []
    for roll1 in range(1, 7):
        for roll2 in range(roll1, 7):
            roll = [roll1, roll2]
            moves, boards = get_valid_moves(player, b, roll)
            if len(boards) == 0:
                if version == 3:
                    equities.append(model.expected_value(model(torch.FloatTensor(np.array(convert_board(invert_board(board) if player ==-1 else board, False, cube=True, RL=True, player=1) for board in boards)))))
                else:
                    equities.append(model.expected_value(model(torch.FloatTensor(np.array([encode_state(invert_board(b) if player == -1 else b, 1)])))).item())
                continue
            if version != 3:
                encoded_boards_tensors = torch.FloatTensor(np.array([encode_state(invert_board(next_board) if player ==-1 else next_board, 1) for next_board in boards]))
            else:
                if player == 1:
                    encoded_boards_tensors = torch.FloatTensor(np.array([convert_board(board, False, cube=True, RL=True, player=1) for board in boards]))
                else:
                    inverted_boards = [invert_board(i) for i in boards]
                    encoded_boards_tensors = torch.FloatTensor(np.array([convert_board(board, False, cube=True, RL=True, player=1) for board in inverted_boards]))
            with torch.no_grad():
                outcome_probs = model(encoded_boards_tensors)
                expected_values = model.expected_value(outcome_probs)
            # Opponent will choose move that benefits them most
            # Make it negative so argmax will work in reinforce_play
            if roll1 != roll2: # Twice probability so append twice
                equities.append(torch.max(expected_values).item())
            equities.append(torch.max(expected_values).item())
    return np.mean(equities)
            
                
   
def reinforce_play(boards, moves, player, ep="self_170000", board=None, lookahead=True):
    """Reinforcement Agent selects a move

    Args:
        boards (list(list(int))): All valid boards
        moves (list(list(tuple))): All valid moves
        player (int): The player whose move it is
        ep (str, optional): The episode chosen to act as the agent. Defaults to "self_170000".
        board (list(int), optional): The current board state. Defaults to None.
        lookahead (bool, optional): Whether or not the agent looks ahead. Defaults to True.

    Returns:
        (moves,boards): The moves and the boards
    """
    model = ReinforceNet3() if ep[0:2] == "V3" else ReinforceNet()
    version = 3 if ep[0:2] == "V3" else 2
    if player == -1:
        inverted_boards = [invert_board(i) for i in boards]
        if ep[0:2] == "V3":
            encoded_boards = [convert_board(board, False, cube=True, RL=True, player=1) for board in inverted_boards]
                
        else:
            encoded_boards = [encode_state(board, -player) for board in inverted_boards]
    else:
        if ep[0:2] == "V3":
            encoded_boards = [convert_board(board, False, cube=True, RL=True, player=1) for board in boards]
        else:
            encoded_boards = [encode_state(board, player) for board in boards]
    
    if ep[0] == "O":
        model.load_state_dict(torch.load(os.path.join("Code","RL","One",f"reinforcement_{ep[3:]}.pth"))['model_state_dict'])
    elif ep[0] == "T":
        model.load_state_dict(torch.load(os.path.join("Code","RL","Two",f"reinforcement_{ep[3:]}.pth"))['model_state_dict'])
    else:
        model.load_state_dict(torch.load(os.path.join("Code","RL",f"reinforcement_{ep}.pth"))['model_state_dict'])
    # encoded_boards = [encode_state(board) for board in inverted_boards]
    board_tensors = torch.FloatTensor(np.array(encoded_boards))
    with torch.no_grad():
        outcome_probs = model(board_tensors)  # Shape: [num_boards, 6]
        expected_values = model.expected_value(outcome_probs)  # Shape: [num_boards]
    
    if not lookahead:
        action_idx = torch.argmax(expected_values).item()
        chosen_board = boards[action_idx]
        chosen_move = moves[action_idx]
    else:
        ### Lookahead ###
        # Find the best equity
        best_equity = torch.max(expected_values).item()

        # Compute threshold
        threshold = best_equity - 0.16 if version != 3 else best_equity - 0.08

        # Keep track of indices for pruning
        pruned_data = [(board, move, equity, idx) for idx, (board, move, equity) in enumerate(zip(boards, moves, expected_values)) if equity >= threshold]

        if not pruned_data:  # Edge case: if all boards are pruned, use the best first-pass move
            action_idx = torch.argmax(expected_values).item()
            return moves[action_idx], boards[action_idx]

        # Extract pruned boards, moves, and original indices
        pruned_boards, pruned_moves, _, pruned_indices = zip(*pruned_data)

        # Apply lookahead on pruned boards
        lookahead_equity = [reinforce_lookahead(pruned_board, -player, model, version) for pruned_board in pruned_boards]

        # Select the best move based on lookahead
        best_idx = torch.argmin(torch.tensor(lookahead_equity)).item()
        # print("lookup equity", lookahead_equity[best_idx])
        chosen_board = pruned_boards[best_idx]
        chosen_move = pruned_moves[best_idx]

    return chosen_move, chosen_board

def load_model(model, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eligibility_traces = checkpoint['eligibility_traces']
    print(f"Model loaded from {path} (episode {checkpoint['episode']})")

# moves, boards = get_valid_moves(1, make_board(), [6,1])
# print(reinforce_play(boards, moves, 1, "self_170000"))
# moves, boards = get_valid_moves(-1, make_board(), [6,1])
# print(reinforce_play(boards, moves, -1, "self_170000"))
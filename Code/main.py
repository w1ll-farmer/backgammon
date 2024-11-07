import pygame
from random_agent import *
from turn import *
import numpy as np

# players = dict()
        
# test = input("Test or Train?\n").lower()
# first_to = 5
# if test != 'test':
#     player_colour = input("Black or White?\n").lower()
#     if player_colour == 'black':
#         players['Black'] = 1
#         players['White'] = -1
#     else:
#         players['White'] = 1
#         players['Black'] = -1

# Each player rolls a die to determine who moves first
black_roll, white_roll = roll_dice()
# Loops in scenario rolls are equal
while black_roll == white_roll:
    black_roll, white_roll = roll_dice()
    
print(f"Black rolled {black_roll}")
print(f"White rolled {white_roll}")
# Black starts first
if black_roll > white_roll:
    print("Black starts")
    player1 = -1
    player2 = 1
else:
    # White starts first
    print("White Starts")
    player1 = 1
    player2 = -1
    
running = True
time_step = 1
board = make_board()
while running:
    if time_step == 1:
        # Initial roll made up of both starting dice
        roll = [black_roll, white_roll]
        moves1, boards1 = get_valid_moves(player1, board, roll)
    else:
        # All other rolls are generated on spot
        roll = roll_dice()
        moves1, boards1 = get_valid_moves(player1, board, roll)
        
    # if player1 == 1:
    print(f"Player {player1} rolled {roll}")
    moves1_stringified = [str(move1) for move1 in moves1]
    move = input("Enter move.")
    while move not in moves1_stringified:
        print("Enter a valid move")
        move = input("")
    # boards1_stringified = [str(board) for board in boards1]
    # print(boards1_stringified)
    move_index = moves1_stringified.index(move)
    board = boards1[move_index]
    print_board(board)
    # Player 2's turn
    roll = roll_dice()
    print(f"Player {player2} rolled {roll}")
    moves2, boards2 = get_valid_moves(player2, board, roll)
    move = []
    while move not in moves2:
        move = []
        for i in range(1+is_double(roll)):
            move.append(generate_random_move())
            move.append(generate_random_move())
    print(f"Move Taken: {move}")
    board = boards2[moves2.index(move)]
    print_board(board)
    # else:
        # print(f"Player {player1} rolled {roll}")
        # move = input("Enter move.")
        # while move not in moves1:
        #     print("Enter a valid move")
        #     move = input("")
        # move_index = board.index(move)
        # board = boards1[move_index]
        # print_board(board)
        # Player
        
        
            

        
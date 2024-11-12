import pygame
from random_agent import *
from turn import *
import numpy as np
from time import sleep 

def start_turn(player, board):
    roll = roll_dice()
    print(f"Player {player} rolled {roll}")
    moves, boards = get_valid_moves(player, board, roll)
    return moves, boards, roll

def human_play(moves, boards):
    moves_stringified = [str(move1) for move1 in moves]
    move = input("Enter move.")
    while move not in moves_stringified:
        print("Enter a valid move")
        move = input("")
    move_index = moves_stringified.index(move)
    board = boards[move_index]
    return board

def randobot_play(roll, moves, boards):
    move = []
    attempts = 0
    
    while move not in moves and attempts < 200000:
        move = []
        for _ in range(1+is_double(roll)):
            move.append(generate_random_move())
            move.append(generate_random_move())
        attempts += 1
        
    if move not in moves:
        move = moves[randint(0, len(moves))]
        
    board = boards[moves.index(move)]
    return board, move

# Each player rolls a die to determine who moves first
black_roll, white_roll = roll_dice()
# Loops in scenario rolls are equal
while black_roll == white_roll:
    black_roll, white_roll = roll_dice()
    
print(f"Black rolled {black_roll}")
print(f"White rolled {white_roll}")
# Black starts first
if black_roll > white_roll:
    print("Computer starts")
    player1 = -1
    player2 = 1
else:
    # White starts first
    print("User Starts")
    player1 = 1
    player2 = -1
    
# running = True
time_step = 1
board = make_board()
while not game_over(board):
    if time_step == 1:
        # Initial roll made up of both starting dice
        roll = [black_roll, white_roll]
        moves1, boards1 = get_valid_moves(player1, board, roll)
        print_board(board)
        print(f"Player {player1} rolled {roll}")
    else:
        # All other rolls are generated on spot
        moves1, boards1, roll = start_turn(player1, board)
    
    sleep(0.5)
    if player1 == 1:
        
        board = human_play(moves1, boards1)
        print_board(board)
        
        # Game ends?
        if is_error(board):
            sleep(2)
            break
        if game_over(board):
            break
        
        sleep(1)
        
        # Player 2's turn
        
        moves2, boards2, roll = start_turn(player2, board)
        board, move = randobot_play(roll, moves2, boards2)
        print(f"Move Taken: {move}")
    else:
        board, move = randobot_play(roll, moves1, boards1)
        print(f"Move Taken: {move}")
        print_board(board)
        sleep(1)
        if is_error(board):
            sleep(2)
            break
        if game_over(board):
            break
        # Player 2 turn
        moves2, boards2, roll = start_turn(player2, board)
        board = human_play(moves2, boards2)
    print_board(board)
    if is_error(board):
        sleep(2)
        break
    time_step +=1
    sleep(1)
        
        
            

        
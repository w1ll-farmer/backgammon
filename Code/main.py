import pygame
from random_agent import *
from turn import *
import numpy as np
from time import sleep 

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
    else:
        # All other rolls are generated on spot
        roll = roll_dice()
        moves1, boards1 = get_valid_moves(player1, board, roll)
        
    if player1 == 1:
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
        if is_error(board):
            sleep(2)
            break
        if game_over(board):
            break
        
        sleep(1)
        
        # Player 2's turn
        
        
        roll = roll_dice()
        print(f"Player {player2} rolled {roll}")
        moves2, boards2 = get_valid_moves(player2, board, roll)
        move = []
        attempts = 0
        
        while move not in moves2 and attempts < 200000:
            move = []
            for i in range(1+is_double(roll)):
                move.append(generate_random_move())
                move.append(generate_random_move())
            attempts += 1
            
        if move not in moves2:
            move = moves2[randint(0, len(moves2))]
            
        print(f"Move Taken: {move}")
        board = boards2[moves2.index(move)]
        
    else:
        # If AI is player 1
        print(f"Player {player1} rolled {roll}")
        moves1, boards1 = get_valid_moves(player1, board, roll)
        move = []
        attempts = 0
        while move not in moves1 and attempts < 200000:
            move = []
            for i in range(1+is_double(roll)):
                move.append(generate_random_move())
                move.append(generate_random_move())
            attempts +=1
            
        if move not in moves2:
            move = moves1[randint(0, len(moves1))]
            
        print(f"Move Taken: {move}")
        board = boards1[moves1.index(move)]
        print_board(board)
        sleep(1)
        if is_error(board):
            sleep(2)
            break
        if game_over(board):
            break
        # Player 2 turn
        roll = roll_dice()
        print(f"Player {player2} rolled {roll}")
        moves2, boards2 = get_valid_moves(player2, board, roll)
        moves2_stringified = [str(move2) for move2 in moves2]
        move = input("Enter move.")
        while move not in moves2_stringified:
            print("Enter a valid move")
            move = input("")
        # boards1_stringified = [str(board) for board in boards1]
        # print(boards1_stringified)
        move_index = moves2_stringified.index(move)
        board = boards2[move_index]
    print_board(board)
    if is_error(board):
        sleep(2)
        break
    time_step +=1
    sleep(1)
        
        
            

        
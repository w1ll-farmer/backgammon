import pygame
from random_agent import *
from turn import *
import numpy as np

players = dict()
        
test = input("Test or Train?\n").lower()
first_to = 5
if test != 'test':
    player_colour = input("Black or White?\n").lower()
    if player_colour == 'black':
        players['Black'] = 1
        players['White'] = -1
    else:
        players['White'] = 1
        players['Black'] = -1

# Each player rolls a die to determine who moves first
black_roll, white_roll = roll_dice()
# Loops in scenario rolls are equal
while black_roll == white_roll:
    black_roll, white_roll = roll_dice()
# Black starts first
if black_roll > white_roll:
    player1 = players['Black']
    player2 = players['White']
else:
    # White starts first
    player1 = players['White']
    player2 = players['Black']
    
running = True
time_step = 1
board = make_board()
while running:
    if time_step == 1:
        roll = [black_roll, white_roll]
        moves1, boards1 = get_valid_moves(player1, board, roll)
    else:
        roll = roll_dice()
        moves1, boards1 = get_valid_moves(player1, board, roll)
    move_dict = dict(zip(moves1, boards1))
    if player1 == 1:
        print(f"You rolled {roll}\n")
        move = input("Enter move.")
        while move not in moves1:
            print("Enter a valid move")
            move = input("")
        board = move_dict[move]
        print_board(board)
        # Player 2's turn
        roll = roll_dice()
        moves2, boards2 = get_valid_moves(player2, board, roll)
        move_dict = dict(zip(moves2, boards2))
        move = []
        while move not in moves2:
            move = []
            for i in range(1+is_double(roll)):
                move.append(generate_random_move(), generate_random_move)
        board = move_dict[move]
        print_board(board)
        
        
            

        
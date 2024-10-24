from random import randint
import numpy as np
import copy
def roll():
    """Simulates the rolling of 2 dice
    
    Returns:
        int: The value of the number rolled on dice 1
        int: The value of the number rolled on dice 2
    """
    return randint(1, 6), randint(1, 6)

def is_blot(dest, opp_colour):
    """Checks if the destination is a blot (if it contains a tile that can be hit)

    Args:
        dest (str): The piece(s) on that point.
        opp_colour (str): The first character of the opponents tile colour

    Returns:
        bool: Whether or not the destination point is a blot
    """
    # If the checker can be hit then it will be a single character, either b or w
    # Which are the two possible options of opp_colour, so a simple == check suffices
    return dest == opp_colour
    
    
def is_double(die1, die2):
    return die1 == die2

def update_board(board, move):
    pass

def must_enter(board, colour):
    if colour > 0 and board[25] > 0:
        return False
    elif colour < 0 and board[24] < 0:
        return False
    else:
        return True
    
    
def can_enter(colour, board, die):
    if colour == -1:
        opp_cords = [i for i in range(0,6)]
        opp_home = board[0:6]
    else:
        opp_cords = [i for i in range(23,17,-1)]
        opp_home = board[18:24][::-1]
    enter = []
    if abs(opp_home[(die)-1]) < 2:
        enter.append(opp_cords[(die)-1])
        # return True
    elif opp_home[(die)-1] * colour > 0:
        # print(opp_home[(die)-1])
        enter.append(opp_cords[(die)-1])
        # return True
    return (24.5+(colour/2),enter)

def get_valid_moves(colour, board, roll):
    moves = []
    boards = []

def get_legal_move(colour, board, die):
    valid_moves = []
    if must_enter(board, colour):
        move = can_enter(colour, board, die)
        if move:
            valid_moves.append(move)
    else:
        if colour == -1:
            possible_starts = [i for i in range(0,24) if board[i] < 0]
            for p in possible_starts:
                if board[p+die] < 2:
                    valid_moves.append((p, p+die))
        else:
            possible_starts = [i for i in range(0,24) if board[i] > 0]
            for p in possible_starts:
                if board[p-die] > 2:
                    valid_moves.append((p, p-die))

def can_bear_off(colour, board):
    pass

def game_over(board):
    return board[24] == 15 or board[25] == -15

def is_gammon(board):
    pass

def is_backgammon(board):
    pass

def turn(colour, board, roll):
    temp_board = copy.deepcopy(board)
    if must_enter(board, colour):
        valid_entries = can_enter(colour, board,roll)
        if  len(valid_entries) < 1:
            print("No valid moves")
            return []
        else:
            # Enter a piece from the bar to the board
            print(f"Valid moves: {valid_entries}")
            move = input("Choose move destination\n")
            while move not in valid_entries:
                print(f"Valid moves: {valid_entries}")
                move = input("Choose move destination\n")
            
            
            

# print(can_enter(1,board = [-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0,0,0,0], roll=[1,2]))
print(game_over([-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0,0,0,0]))

"""
roll_dice()
repeat getlegamoves once a move is confirmed
getlegalmoves(colour, board, roll) calls getlegalmove(colour, board, die) for each die
    valid = []
    if must_enter(colour, board):
        if can_enter(colour, board, roll):
            valid += [bar, point]
        return valid
    else:
        for point occupied by player's checker
            if valid move?
                valid += [start_point, end_point]
            
            
valid move:
    if all checkers home:
        if die = exact distance to move off board, bear off
        if furthest occupied point not as far as die roll: bear off checker on furthest point 
        otherwise skip
At end of game check for gammon:
    if sum([beard off]) == 0: points = 2
backgammon too

also give opportunity for doubling cube throughout: implement after core engine
    """
    
    
    
    
    



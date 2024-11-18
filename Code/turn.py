from random import randint
import numpy as np
import copy
import pandas as pd


def make_board():
    return [
        -2,0,0,0,0,5,  0,3,0,0,0,-5,
         5,0,0,0,-3,0, -5,0,0,0,0,2,
         0,0,0,0
         ]

def print_board(board):
    print("Board:")
    print(board[0:6],'\t',board[6:12])
    print(board[12:18],'\t',board[18:24])
    print(board[24:])
    # for i in range(len(board)):
    #     print(board[])

def roll_dice():
    # Returns the value of the two dice rolled
    return randint(1, 6), randint(1, 6)
    
def is_double(roll):
    # Checks if two identical dice are rolled
    return roll[0] == roll[1]


def update_board(board, move):
    """Updates the board after a move is made

    Args:
        board ([int]): The representation of the board
        move ((int, int)): The start position and end position of a move

    Returns:
        [int]: The board after the move has been played
    """
    start, end = move
    # Copies by value
    board_copy = board.copy()
    
    if board_copy[start] > 0:
        # Move piece away from point
        board_copy[start] -=1
        if board_copy[end] == -1:
            # Hit a piece off the board
            board_copy[end] = 1
            board_copy[24] -= 1
        else:
            board_copy[end] += 1
            
    else:
        # Move piece away from point
        board_copy[start] += 1
        if board_copy[end] == 1:
            # Hit a piece off the board
            board_copy[end] = -1
            board_copy[25] += 1
        else:
            board_copy[end] -= 1
    return board_copy

def get_home_info(player):
    if player == 1:
        cords = [i for i in range(0,6)]
        home = board[0:6]
    else:
        cords = [i for i in range(23,17,-1)]
        home = board[18:24][::-1]
    return cords, home

def must_enter(board, colour):
    """Checks if the player has a checker on the bar

    Args:
        board ([int]): The representation of the board
        colour (int): Whether the checker is white (1) or black (-1)

    Returns:
        bool: Whether the player has a checker on the bar
    """
    if colour > 0 and board[25] > 0:
        # 25 is bar for player 1
        return True
    elif colour < 0 and board[24] < 0:
        # 24 is bar for player -1
        return True
    else:
        return False
    
    
def can_enter(colour, board, die):
    """Checks for a valid move from the bar to the board

    Args:
        colour (int): Whether the checker is white (1) or black (-1)
        board ([int]): The representation of the board
        die (int): The value of the roll of one of the dice

    Returns:
        (int, int): The start and end points of the valid move
    """
    opp_cords, opp_home = get_home_info(-colour)
    enter = 0
    if abs(opp_home[(die)-1]) < 2:
        enter = (opp_cords[(die)-1])
        # return True
    elif opp_home[(die)-1] * colour > 0:
        # print(opp_home[(die)-1])
        enter = (opp_cords[(die)-1])
        # return True
    return (int(24.5+(colour/2)),enter)

def all_checkers_home(colour, board):
    """Checks all checker are home so they can be beard off

    Args:
        colour (int): Checker's colour. -1 for black 1 for white
        board ([int]): The representation of the board

    Returns:
        bool: Whether or not all checkers are home
    """
    if colour == -1:
        if len([i for i in board[0:18] if i < 0]) == 0:
            return True
    else:
        if len([i for i in board[6:24] if i > 0]) == 0:
            return True
    return False

def get_legal_move(colour, board, die):
    """Identifies all valid moves for a single die roll

    Args:
        colour (int): The checker's colour. 1 for white -1 for black
        board ([int]): The representation of the board
        die (int): The value of the roll of one of the dice

    Returns:
        [(int)]: Start and end point pairs for each valid move
    """
    valid_moves = []
    # If the player has a checker on the bar
    if must_enter(board, colour):
        # print(f"Must Enter {die}")
        move = can_enter(colour, board, die)
        if move:
            valid_moves.append(move)
    else:
        if colour == -1: # Black player's move
            if all_checkers_home(colour, board):
                print(f"All home")
                # Can a piece be beard off directly?
                if board[24-die] < 0:
                    print(f"Bearing off {24-die, die}")
                    valid_moves.append((24-die, 26))
                    
                # Can a piece be beard off due to all checkers being closer than die
                elif not game_over(board):
                    furthest_back = 23
                    found = False
                    i = 18
                    
                    # Check for furthest back piece
                    while i < 24 and not found:
                        if board[i] < 0:
                            furthest_back = i
                            found=True
                        i +=1
                        
                    # If the die roll is greater than furthest back occupied point
                    # Then a checker on that point can be beard off
                    if die > 24-furthest_back:
                        valid_moves.append((furthest_back, 26))
                        
            else:
                possible_starts = [i for i in range(0,24) if board[i] < 0]
                # print(possible_starts)
                for p in possible_starts:
                    if p+die < 24:
                        if board[p+die] < 2:
                            valid_moves.append((p, p+die))
                        
        else: # White player's move
            if all_checkers_home(colour, board):
                if board[die] > 0:
                    valid_moves.append((die, 27))
                    
                elif not game_over(board):
                    furthest_back = 0
                    found = False
                    i = 5
                    
                    while i > -1 and not found:
                        if board[i] > 0:
                            furthest_back = i
                            found = True
                        i -= 1
                        
                    if die > furthest_back:
                        valid_moves.append((furthest_back, 27))
                        
            else:
                possible_starts = [i for i in range(0,24) if board[i] > 0]
                # print(possible_starts)
                for p in possible_starts:
                    if p - die >= 0:
                        if board[p-die] > -2:
                            valid_moves.append((p, p-die))
    return valid_moves


# def get_valid_moves(colour, board, roll):
#     # make sure check for doubles so player gets four moves
#     moves = []
#     boards = []
#     possible_moves = [[],[],[],[]]
#     # Using first die 
#     possible_moves[0] = get_legal_move(colour, board, roll[0])
#     for move1 in possible_moves[0]:
#         # Generate a temporary board if any possible move is taken
#         temp_board = update_board(board, move1)
#         print_board(temp_board)
#         print(move1)
#         # Generate all possible second moves based on each temporary board
#         possible_moves[1] = get_legal_move(colour, temp_board, roll[1])
        
#         # If they can only move once
#         if len(possible_moves[1]) == 0:
#             moves.append([move1])
#             boards.append(update_board(temp_board, move1))
#             print('Only move once')
#         else:    
#             for move2 in possible_moves[1]:
#                 print_board(temp_board)
#                 # print(move1,move2)
#                 temp_board = update_board(temp_board, move2)
#                 if not is_double(roll):
#                     # If only two die (not a double, append all possible pairs of moves)
#                     moves.append([move1, move2])
#                     boards.append(temp_board)
#                     print('Not double')
#                 else:
#                     # If a double is rolled, they get 4 moves, rather than 2
#                     # Generate another temporary board based on first 2 moves
                    
#                     # print_board(temp_board)
#                     # print(move1, move2,"\n")
#                     # Generate all possible 3rd moves
#                     possible_moves[2] = get_legal_move(colour, temp_board, roll[0])
                    
#                     # In case the player can only use 2/4 of their rolls
#                     if len(possible_moves[2]) == 0:
#                         moves.append([move1, move2])
#                         boards.append(temp_board)
#                         print('Only move twice')
#                     else:
#                         for move3 in possible_moves[2]:
#                             # Generate all possible boards based on first 3 moves
#                             temp_board = update_board(temp_board, move3)
#                             # Enumerate all possible fourth moves
#                             possible_moves[3] = get_legal_move(colour, temp_board, roll[1])
                            
#                             # In case player can only use 3/4 of their rolls
#                             if len(possible_moves[3]) == 0:
#                                 moves.append([move1, move2, move3])
#                                 boards.append(temp_board)
#                                 print('Only move 3')
#                             else:
#                                 for move4 in possible_moves[3]:
#                                     # Append all possible sets of moves
#                                     moves.append([move1,move2,move3,move4])
#                                     boards.append(update_board(temp_board, move4))
#                                     # print(move1,move2,move3,move4)
#                                     # print_board(update_board(temp_board, move4))
#     # In case there are moves that are only possible if die2 is used first
#     # If dice are same then no need to swap order
#     if not is_double(roll): 
#         print("Not a double")
#         possible_moves[0] = get_legal_move(colour, board, roll[1])
#         for move1 in possible_moves[0]:
#             # Get the resulting board of a potential move
#             temp_board = update_board(board, move1)
#             # Generate all possible second moves based on each first move
#             possible_moves[1] = get_legal_move(colour, temp_board, roll[0])
            
#             # If only one move can be used
#             if len(possible_moves[1]) == 0:
#                 moves.append([move1])
#                 boards.append(temp_board)
                
#             for move2 in possible_moves[1]:
#                 moves.append([move1, move2])
#                 boards.append(update_board(temp_board, move2))
        
#     return moves, boards

def get_valid_moves(colour, board, roll):
    moves = []
    boards = []
    possible_moves = [[], [], [], []]

    # Use the first die in the roll
    possible_moves[0] = get_legal_move(colour, board, roll[0])
    
    for move1 in possible_moves[0]:
        temp_board1 = update_board(board, move1)  # Apply first move
        possible_moves[1] = get_legal_move(colour, temp_board1, roll[1])

        # If only one move can be used
        if len(possible_moves[1]) == 0:
            moves.append([move1])
            boards.append(temp_board1)
        else:
            for move2 in possible_moves[1]:
                temp_board2 = update_board(temp_board1, move2)  # Apply second move
                
                if not is_double(roll):
                    # If not a double, append pairs of moves
                    moves.append([move1, move2])
                    boards.append(temp_board2)
                else:
                    # For doubles, attempt up to four moves
                    possible_moves[2] = get_legal_move(colour, temp_board2, roll[0])
                    
                    if len(possible_moves[2]) == 0:
                        moves.append([move1, move2])
                        boards.append(temp_board2)
                    else:
                        for move3 in possible_moves[2]:
                            temp_board3 = update_board(temp_board2, move3)  # Apply third move
                            possible_moves[3] = get_legal_move(colour, temp_board3, roll[1])
                            
                            if len(possible_moves[3]) == 0:
                                moves.append([move1, move2, move3])
                                boards.append(temp_board3)
                            else:
                                for move4 in possible_moves[3]:
                                    final_board = update_board(temp_board3, move4)  # Apply fourth move
                                    moves.append([move1, move2, move3, move4])
                                    boards.append(final_board)

    # Handle the reverse order of dice rolls if not a double
    if not is_double(roll): 
        possible_moves[0] = get_legal_move(colour, board, roll[1])
        for move1 in possible_moves[0]:
            temp_board1 = update_board(board, move1)
            possible_moves[1] = get_legal_move(colour, temp_board1, roll[0])

            if len(possible_moves[1]) == 0:
                moves.append([move1])
                boards.append(temp_board1)
            else:
                for move2 in possible_moves[1]:
                    final_board = update_board(temp_board1, move2)
                    moves.append([move1, move2])
                    boards.append(final_board)

    return moves, boards


board = make_board()
# moves, boards = get_valid_moves(-1, board, [1,1])
# for i in range(len(boards)):
#     print_board(boards[i])
#     print(moves[i])
# print_board(board)

def game_over(board):
    # Checks if the game is over
    return board[27] == 15 or board[26] == -15

def is_backgammon(board):
    # Checks for a backgammon
    return sum(board[18:24]) + board[25] > 0 or sum(board[0:6]) + board[24] < 0

def is_gammon(board):
    # Checks for a gammon
    return board[26] == 0 or board[27] == 0

def is_error(board):
    if sum([i for i in board if i < 0]) != -15 or sum([i for i in board if i > 0]) != 15:
        print(sum([i for i in board if i < 0]),sum([i for i in board if i > 0]))
    else:
        return False

    
    
    
    
    



from random import randint
import numpy as np
import copy
import pandas as pd
from constants import *
from time import sleep
if GUI_FLAG:
    import pygame
    pygame.init()
    
def make_board():
    return [
        -2,0,0,0,0,5,  0,3,0,0,0,-5,
         5,0,0,0,-3,0, -5,0,0,0,0,2,
         0,0,0,0
         ]

def print_board(board):
    if commentary:
        print("Board:")
        print(board[0:6],'\t',board[6:12])
        print(board[12:18],'\t',board[18:24])
        print(board[24:])

def roll_dice():
    # Returns the value of the two dice rolled
    return randint(1, 6), randint(1, 6)
    # return 1, 5
    """return 5, 1 !!! Unsymmetrical AI decisions !!!
    note if its 5, 1 that white wins 25-0, and if other way round causes a loop due to 
    constant hitting, entering and hitting cycle"""
    
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
        player = 1
    else:
        player = -1
    
    board_copy[start] -= player
    if board_copy[end] == -player:
        board_copy[end] = player
        board_copy[int(24.5-(player/2))] -= player
    else:
        board_copy[end] += player
        
    return board_copy

def all_past(board):
    """Checks that all pieces have passed each other

    Args:
        board (list(int)): Board representation

    Returns:
        bool: Whether the pieces are all passed each other
    """
    if board[24] < 0 or board[25] > 0:
        return False
    
    furthest_back_white = 23
    furthest_back_black = 0
    
    while board[furthest_back_white] < 1:
        furthest_back_white-=1
    while board[furthest_back_black] > -1:
        furthest_back_black +=1
        
    if furthest_back_black > furthest_back_white:
        return True
    else:
        return False

def get_home_info(player, board):
    """Returns the cords and state of player's home board

    Args:
        player (int): -1 for black, 1 for white
        board (list(int)): Board representation

    Returns:
        list(int), list(int): cords and state of player's home
    """
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
    opp_cords, opp_home = get_home_info(-colour, board)
    enter = 0
    if colour == 1 and opp_home[die-1] > -2:
        enter = opp_cords[(die)-1]
        return (25, enter)
    elif colour == -1 and opp_home[die-1] < 2:
        enter = opp_cords[die-1]
        return (24, enter)
    return False

def all_checkers_home(colour, board):
    """Checks all checker are home so they can be borne off

    Args:
        colour (int): Checker's colour. -1 for black 1 for white
        board ([int]): The representation of the board

    Returns:
        bool: Whether or not all checkers are home
    """
    if colour == -1:
        if len([i for i in board[0:18] if i < 0]) == 0 and board[24] >= 0:
            return True
    else:
        if len([i for i in board[6:24] if i > 0]) == 0 and board[25] <= 0:
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
            if commentary:
                print("Player must enter but cannot")
    else:
        if colour == -1: # Black player's move
            if all_checkers_home(colour, board):
                if commentary:
                    print(f"All home")
                # Can a piece be borne off directly?
                if board[24-die] < 0:
                    if commentary:
                        print(f"Bearing off {24-die, die}")
                    valid_moves.append((24-die, 26))
                    
                # Can a piece be borne off due to all checkers being closer than die
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
                    # Then a checker on that point can be borne off
                    if die > 24-furthest_back:
                        valid_moves.append((furthest_back, 26))
            
            # All points occupied by the -1 player
            possible_starts = [i for i in range(0,24) if board[i] < 0]
            for p in possible_starts:
                # If ending location is on the board
                if p+die < 24:
                    # If ending location is occupied by black or a white blot
                    if board[p+die] < 2:
                        valid_moves.append((p, p+die))
                        
        else: # White player's move
            if all_checkers_home(colour, board):
                # Bear off directly
                if board[die-1] > 0:
                    valid_moves.append((die-1, 27))
                    
                elif not game_over(board):
                    # Furthest piece back is less than dice roll so can be borne off too
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
                        
            # Identify all points occupied by 1 player
            possible_starts = [i for i in range(0,24) if board[i] > 0]
            # If start + roll is on the board
            for p in possible_starts:
                if p - die >= 0:
                    # If the piece is occupied by white or is a black blot
                    if board[p-die] > -2:
                        valid_moves.append((p, p-die))
    return valid_moves

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
                    
                    # If no more moves can be made
                    if len(possible_moves[2]) == 0:
                        moves.append([move1, move2])
                        boards.append(temp_board2)
                    else:
                        for move3 in possible_moves[2]:
                            temp_board3 = update_board(temp_board2, move3)  # Apply third move
                            possible_moves[3] = get_legal_move(colour, temp_board3, roll[1])
                            
                            # If no more moves can be made
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
    # Checks the right number of pieces are on the board
    if sum([i for i in board if i < 0]) != -15 or sum([i for i in board if i > 0]) != 15:
        print(sum([i for i in board if i < 0]),sum([i for i in board if i > 0]))
        errorFile = open('Error.txt','a')
        errorFile.write(f"Board: {board}\n")
        sleep(10)
        return True
    else:
        return False


def calc_pips(board, player):
    if player == 1:
        start_positions = [point for point in range(len(board)) if board[point] > player]
    else:
        start_positions = [point for point in range(len(board)) if board[point] < player]
    total = 0
    for point in start_positions:
        if player == 1:
            if point == 27:
                continue
            elif point == 25:
                total += abs(25*board[point])
            else:
                total += abs((point + 1)*board[point])
        else:
            if point == 26:
                continue
            elif point == 24:
                total += abs(25*board[point])
            else:
                total += abs((24 - point)*board[point])
    return total

def count_blots(board, player):
    """Counts the number of a player's blots in a region

    Args:
        board (list(int)): Specified board region
        player (int): Whether player is controlling black or white

    Returns:
        int: Number of blots in specified region
    """
    return len([i for i in board if i == player])

def count_walls(board, player):
    """Counts the number of walls in a specified region

    Args:
        board (list(int)): Specified region checking in
        player (int): Whether player is controlling black or white

    Returns:
        int: Numbers of player's walls in region
    """
    if player == 1:
        return len([i for i in board if i > player])
    else:
        return len([i for i in board if i < player])

def is_wall(point, player):
    """Checks if a point is a wall occupied by current player

    Args:
        point (int): The pieces occupying a point on the board
        player (int): -1 for black, 1 for white

    Returns:
        Bool: True if wall, False if not
    """
    if player == 1:
        return point > player
    else:
        return point < player
    
def get_furthest_back(board, player):
    """Identifies the player's furthest back piece

    Args:
        board (list(int)): The representation of the board
        player (int): -1 for black, 1 for white

    Returns:
        int: The point that the furthest back piece occupies
    """
    if player == 1:
        furthest_back = 23
        while board[furthest_back] < player:
            furthest_back -=1
    else:
        furthest_back = 0
        while board[furthest_back] > player:
            furthest_back += 1
    return furthest_back

def did_move_piece(point_before, point_after, player):
    """Checks if a piece was moved in the turn

    Args:
        point_before (int): The number of pieces occupied the point before the move was made
        point_after (int): The number of pieces occupying the point after the move has been made
        player (int): -1 for black, 1 for white

    Returns:
        Bool: True if a piece on the point was moved, else False
    """
    if player == 1 and point_before > point_after:
        return True
    elif player == -1 and point_before < point_after:
        return True
    else:
        return False

def calc_prime(board, player):
    prime = 0
    max_prime = 0
    for point in board:
        if is_wall(point, player):
            prime +=1
        else:
            if prime > max_prime: max_prime = prime
            prime = 0
        
    if prime > max_prime: max_prime = prime
    return max_prime
    
def prob_opponent_can_hit(player, board, point):
    start_points = [i for i in range(len(board)) if (player == 1 and board[i] < 0) or (player == -1 and board[i] > 0)]
    can_hit = 0
    found = []
    for roll1 in range(1,7):
        for roll2 in range(1, 7):
            # i = 0
            # found = 0
            for s in start_points:
                if f"{roll1,roll2}" not in found:
                    if (s + player*roll1 == point or s + player*roll2 == point):
                        can_hit +=1
                        found.append(f"{roll1,roll2}")
                        
                    elif roll1 == roll2 and can_double_hit(player, point, s, roll1, roll2, board):
                        can_hit +=1
                        found.append(f"{roll1,roll2}")
                        
                    elif player == 1 and roll1 != roll2:
                        if s+roll1+roll2 == point:
                            if board[s + roll1] <= 1 or board[s + roll2] <= 1: 
                                can_hit += 1
                                found.append(f"{roll1,roll2}")
                                
                    elif player == -1 and roll1 != roll2:
                        if s-roll1-roll2 == point:
                            if board[s - roll1] >= -1 or board[s - roll2] >= -1: 
                                can_hit += 1
                                found.append(f"{roll1,roll2}")          
    return can_hit/36



def can_double_hit(player, point, s, roll1, roll2, board):
    if player ==1:
        possible = False
        moved = 0
        while moved < 4 and not possible:
            moved += 1
            if s + roll1*moved == point:
                possible = True
            if s+ roll1*moved > 23:
                break
            if board[s+ roll1*moved] < -1:
                break
                
        return possible
            
    else:
        possible = False
        moved = 0
        while moved < 4 and not possible:
            moved += 1
            if s + roll1*-moved == point:
                possible = True
            if s+ roll1*-moved < 0:
                break
            if board[s+ roll1*-moved] > 1:
                break
                
        return possible

def calc_blockade_pass_chance(board, player):
    passed = 0
    furthest = get_furthest_back(board, -player)
    if player == 1:
        loc = 0
        while board[loc] < player and loc < 24:
            loc +=1
        block_start=loc
        while board[loc+1] >= player:
            loc += 1
    else:
        loc = 23
        while board[loc] > player and loc > -1:
            loc -= 1
        block_start=loc
        while board[loc-1] <= player:
            loc -= 1
    
    for roll1 in range(1,7):
        for roll2 in range(1, 7):
            if player == -1:
                if furthest+(roll1+roll2)*(1+roll2==roll1) > loc:
                    if is_double([roll1, roll2]):
                        possible = True
                        moved = 0
                        while moved < 4 and possible:
                            moved += 1
                            if furthest + roll1*moved >= block_start and \
                                furthest+roll1*moved <= loc:
                                possible = False
                            if furthest + roll1*moved > 23 and furthest + roll1*moved != 27:
                                possible = False
                            if board[furthest+ roll1*moved] < -1:
                                possible = False
                        if possible:
                            passed += 1
                    
                    elif furthest + roll1 < block_start or \
                            furthest + roll2 < block_start:
                            passed += 1
                            
                    elif furthest + roll1 > loc or furthest + roll2 > loc:
                        passed += 1
            else:
                if furthest-(roll1+roll2)*(1+roll2==roll1) > loc:
                    if is_double([roll1, roll2]):
                        possible = True
                        moved = 0
                        while moved < 4 and possible:
                            moved += 1
                            if furthest - roll1*moved <= block_start and \
                                furthest-roll1*moved >= loc:
                                possible = False
                            if furthest - roll1*moved < 0:
                                possible = False
                            if board[furthest- roll1*moved] > 1:
                                possible = False
                        if possible:
                            passed += 1
                    
                    elif furthest - roll1 > block_start or \
                            furthest - roll2 > block_start:
                            passed += 1
                            
                    elif furthest - roll1 < loc or furthest - roll2 < loc:
                        passed += 1
    return passed/36

def decimal_to_binary(decimal):
    binary = [0]*7
    for i in range(6,-1,-1):
        if decimal >= 2**i:
            binary[6-i] = 1
            decimal -= 2**i
    return binary

def convert_point(point):
    base = [0]* 10
    if point < 0:
        for i in range(0, 5):
            if point <= i-5 and (i == 0 or i == 3) or point == i-5:
                base[i] = 1
    elif point > 0:
        for i in range(5, 10):
            if point >= i - 4 and (i == 9 or i == 6) or point == i-4:
                base[i] = 1
    return base


def convert_bar(point):
    base = [0]*3
    if point < 0:
        for i in range(0, 2):
            if point <= i-2 and i ==0 or point == i-2:
                base[i]=1
    elif point > 0:
        for i in range(0, 3):
            if point >= i and i == 2 or point == i:
                base[i]=1
    return base

def list_to_str(lst, commas=True,spaces=True):
    if not commas:
        str_board = ""
        for i in lst:
            if spaces:
                str_board += f"{i} "
            else:
                str_board += f"{i}"
    else:
        lst = str(lst)[1:-1]
    return lst if commas else str_board


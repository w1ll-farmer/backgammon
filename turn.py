from random import randint
import numpy as np
import copy
board = [-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0,0,0,0]
def roll():
    # Returns the value of the two dice rolled
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
    # Checks if two identical dice are rolled
    return die1 == die2


def update_board(board, move):
    """Updates the board after a move is made

    Args:
        board ([int]): The representation of the board
        move ((int, int)): The start position and end position of a move

    Returns:
        [int]: The board after the move has been played
    """
    start, end = move
    board_copy = copy.deepcopy(board)
    if board_copy[start] > 0:
        board_copy[start] -=1
        if board_copy[end] == -1:
            board_copy[end] = 1
        else:
            board_copy[end] += 1
    else:
        board_copy[start] += 1
        if board_copy[end] == 1:
            board_copy[end] = -1
        else:
            board_copy[end] -= 1
    return board_copy


def must_enter(board, colour):
    """Checks if the player has a checker on the bar

    Args:
        board ([int]): The representation of the board
        colour (int): Whether the checker is white (1) or black (-1)

    Returns:
        bool: Whether the player has a checker on the bar
    """
    if colour > 0 and board[25] > 0:
        return True
    elif colour < 0 and board[24] < 0:
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
    if colour == -1:
        opp_cords = [i for i in range(0,6)]
        opp_home = board[0:6]
    else:
        opp_cords = [i for i in range(23,17,-1)]
        opp_home = board[18:24][::-1]
    enter = 0
    if abs(opp_home[(die)-1]) < 2:
        enter = (opp_cords[(die)-1])
        # return True
    elif opp_home[(die)-1] * colour > 0:
        # print(opp_home[(die)-1])
        enter = (opp_cords[(die)-1])
        # return True
    return (24.5+(colour/2),enter)

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


def get_valid_moves(colour, board, roll):
    # make sure check for doubles so player gets four moves
    possible_first_moves = get_legal_move(colour, board, roll[0])
    for move in possible_first_moves:
        temp_board = update_board(board, move)
        # possible_second_moves

    
    
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
        move = can_enter(colour, board, die)
        if move:
            valid_moves.append(move)
    else:
        if colour == -1: # Black player's move
            if all_checkers_home(colour, board):
                # Can a piece be beard off directly?
                if board[24-die] < 0:
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
                        valid_moves.append((24-furthest_back, 26))
                        
            else:
                possible_starts = [i for i in range(0,24) if board[i] < 0]
                # print(possible_starts)
                for p in possible_starts:
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
                    if board[p-die] > -2:
                        valid_moves.append((p, p-die))
    return valid_moves


def game_over(board):
    # Checks if the game is over
    return board[27] == 15 or board[26] == -15

def is_backgammon(board):
    # Checks for a backgammon
    return sum(board[18:24]) + board[25] > 0 or sum(board[0:6]) + board[24] < 0

def is_gammon(board):
    # Checks for a gammon
    return board[26] == 0 or board[27] == 0

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
# print(game_over([-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0,0,0,0]))

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
    
    
    
    
    



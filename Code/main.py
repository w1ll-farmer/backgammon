import pygame
import numpy as np
from pygame.locals import *
from random_agent import *
from turn import *
from time import sleep 
from greedy_agent import *
from constants import *
from gui import *
              
def start_turn(player, board):
    """Rolls the dice and finds all possible moves.

    Args:
        player (int): The encoding of the player, -1 or 1
        board ([int]): The representation of the board

    Returns:
        ([(int, int)], [int], [int]): Possible moves, associated boards, diceroll
    """
    roll = roll_dice()
    if commentary:
        print(f"Player {player} rolled {roll}")
    moves, boards = get_valid_moves(player, board, roll)
    return moves, boards, roll

def human_play(moves, boards):
    """Lets the human player make a move

    Args:
        moves ([(int, int)]): Possible start-end pairs.
        boards ([[int]]): All boards associated with each move

    Returns:
        [int]: The resulting board after making the move
    """
    if len(moves) > 0:
        moves_stringified = [str(move1) for move1 in moves]
        move = input("Enter move.")
        # Loop until valid move is selected
        while move not in moves_stringified:
            print("Enter a valid move")
            move = input("")
        move_index = moves_stringified.index(move)
        board = boards[move_index]
    else:
        print("No valid moves available")
    return board

def randobot_play(roll, moves, boards):
    """Random agent makes a move

    Args:
        roll ([int]): The dice roll.
        moves ([[(int, int)]]): List of all possible start-end pairs
        boards ([[int]]): The boards associated to each move

    Returns:
        [int], [(int, int)]: The resulting board and move chosen
    """
    move = []
    attempts = 0
    # Repeats until 200,000 random moves have been chosen
    # Or until a valid move has been selected
    while move not in moves and attempts < 200000:
        move = []
        for _ in range(1+is_double(roll)):
            move.append(generate_random_move())
            move.append(generate_random_move())
        attempts += 1
    # In case no random move was valid
    if attempts == 200000:
        if commentary:
            print('Randobot cannot find moves')
    if move not in moves:
        if len(moves) > 1:
            move = moves[randint(0, len(moves)-1)]
        elif len(moves) == 1:
            move = moves[0]
    
        
    board = boards[moves.index(move)]
    return board, move

def greedy_play(moves, boards, current_board, player):
    """Greedy agent makes a move

    Args:
        moves ([(int, int)]): Start-end pairs of all valid moves
        boards ([int]): Board representation
        current_board ([int]): Board before greedy plays
        player (int): The player making the move

    Returns:
        [int]: The board resulting from move made
    """
    scores = [evaluate(moves[i], current_board, boards[i], player) for i in range(len(moves))]
    sorted_triplets = sorted(zip(scores, boards, moves), key=lambda x: x[0], reverse=True)
    sorted_scores, sorted_boards, sorted_moves = zip(*sorted_triplets)
    if commentary:
        print(f"Player {player} played {sorted_moves[0][0]}, {sorted_moves[0][1]}")
    return list(sorted_boards)[0]
    

###############
## MAIN BODY ##
###############
def backgammon(games=1):
    if GUI_FLAG == True:
        background_board = Background('Images/board_unaltered.png')
        white_score = Shape('Images/White-score.png', 38, SCREEN_HEIGHT-150)
        black_score = Shape('Images/Black-score.png', 40, 150)
        white_checker = Shape('Images/black_pawn.png', 200, 200, 42, 42)
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
            pygame.display.update()
            framesPerSec.tick(30)
            background_board.render()
            
            white_score.draw(window)
            white_score.addText(window, '0/5',black)
            
            black_score.draw(window)
            black_score.addText(window, '0/5',white)
            white_checker.draw(window)
            
    p1vector = [0,0,0] #win, gammon win, backgammon win
    pminus1vector = [0,0,0] #win, gammon win, backgammon win
    
    for _ in range(games):
        time_step = 1
    # Each player rolls a die to determine who moves first
        black_roll, white_roll = roll_dice()
        # Loops in scenario rolls are equal
        while black_roll == white_roll:
            black_roll, white_roll = roll_dice()
        if commentary:
            print(f"Black rolled {black_roll}")
            print(f"White rolled {white_roll}")
        # Black starts first
        if black_roll > white_roll:
            if commentary:
                print("Computer starts")
            player1 = -1
            player2 = 1
        else:
            # White starts first
            if commentary:
                print("User Starts")
            player1 = 1
            player2 = -1
        # running = True
        board = make_board()
        while not game_over(board) and not is_error(board):
            print(time_step)
            if time_step == 1:
                # Initial roll made up of both starting dice
                roll = [black_roll, white_roll]
                moves1, boards1 = get_valid_moves(player1, board, roll)
                print_board(board)
                if commentary:
                    print(f"Player {player1} rolled {roll}")
            else:
                # All other rolls are generated on spot
                moves1, boards1, roll = start_turn(player1, board)
            if USER_PLAY:
                sleep(0.5)
            if player1 == 1:
                if len(moves1) > 0:
                    if USER_PLAY:
                        board = human_play(moves1, boards1)
                    else:
                        board, move = randobot_play(roll, moves1, boards1)
                        if commentary:
                            print(f"Move Taken: {move}")
                else:
                    if commentary:
                        print("No move can be played")
                print_board(board)
                
                # Game ends?
                if is_error(board):
                    sleep(10)
                    break
                if game_over(board):
                    break
                if USER_PLAY:
                    sleep(1)
                
                # Player 2's turn
                
                moves2, boards2, roll = start_turn(player2, board)
                if len(moves2) > 0:
                    board, move = randobot_play(roll, moves2, boards2)
                    if commentary:
                        print(f"Move Taken: {move}")
                else:
                    if commentary:
                        print("No move can be played")
            else:
                if len(moves1) > 0:
                    board, move = randobot_play(roll, moves1, boards1)
                    if commentary:    
                        print(f"Move Taken: {move}")
                else:
                    if commentary:
                        print("No move can be played")
                print_board(board)
                if USER_PLAY:
                    sleep(1)
                if is_error(board):
                    sleep(10)
                    break
                if game_over(board):
                    break
                # Player 2 turn
                moves2, boards2, roll = start_turn(player2, board)
                if len(moves2) > 0:
                    if USER_PLAY:
                        board = human_play(moves2, boards2)
                    else:
                        board, move = randobot_play(roll, moves2, boards2)
                        if commentary:
                            print(f"Move Taken: {move}")
                else:
                    if commentary:
                        print("No move can be played")
            print_board(board)
            if is_error(board):
                sleep(10)
                break
            time_step +=1
            if USER_PLAY:
                sleep(1)
        if game_over(board):
            if commentary:
                print("GAME OVER")
            if is_backgammon(board):
                if board[26] == -15:
                    pminus1vector[2] +=1
                    if commentary:
                        print("Player -1 win")
                else:
                    p1vector[2] +=1
                    if commentary:
                        print("Player 1 win")
                if commentary:
                    print("By backgammon")
            elif is_gammon(board):
                if board[26] == -15:
                    pminus1vector[1] +=1
                    if commentary:
                        print("Player -1 win")
                else:
                    p1vector[1] +=1
                    if commentary:
                        print("Player 1 win")
                if commentary:
                    print("By gammon")
            else:
                if board[26] == -15:
                    pminus1vector[0] +=1
                    if commentary:
                        print("Player -1 win")
                else:
                    p1vector[0] +=1
                    if commentary:
                        print("Player 1 win")
    return p1vector, pminus1vector
            
            
if __name__ == "__main__":
    games = 5      
    p1vector, pminus1vector = backgammon(games)
    print(p1vector,pminus1vector)
            
            
                

            
import pygame
import sys
import numpy as np
from pygame.locals import *
from random_agent import *
from time import sleep, time
from random import randint, uniform
import torch
import os

from turn import *
from greedy_agent import *
from constants import *
from gui import *
from testfile import *
from data import *
from genetic_agent import *
from expectimax_agent import *
from adaptive_agent import *
from double import *
from make_database import *
from deep_agent import *
from gnubg_interact import *

global background
global white_score
global black_score
global w_score 
global b_score

### INITIALISE BOARD ###
if GUI_FLAG:
    background = Background('Images/two_players_back.png')
    white_score = Shape('Images/White-score.png', SCREEN_WIDTH-36, SCREEN_HEIGHT//2 + 40)
    black_score = Shape('Images/Black-score.png', SCREEN_WIDTH-35, SCREEN_HEIGHT//2 - 40)
    w_score, b_score = 0,0
    
def start_turn(player, board):
    """Rolls the dice and finds all possible moves.

    Args:
        player (int): The encoding of the player, -1 or 1
        board ([int]): The representation of the board

    Returns:
        ([(int, int)], [int], [int]): Possible moves, associated boards, diceroll
    """
    roll = roll_dice()
    if GUI_FLAG:
        roll = display_dice_roll(player)
    if commentary:
        print(f"Player {player} rolled {roll}")
    moves, boards = get_valid_moves(player, board, roll)
    if test:
        check_moves(board, boards, player, roll)
    return moves, boards, roll

##########################
## START OF HUMAN PLAY ##
#########################

def human_play(moves, boards, start_board, roll, colour):
    """Lets the human player make a move

    Args:
        moves ([(int, int)]): Possible start-end pairs.
        boards ([[int]]): All boards associated with each move

    Returns:
        [int]: The resulting board after making the move
    """
    if len(moves) > 0:
        if not GUI_FLAG:
            #### IF NOT USING GUI ####
            #### IGNORE FOR GUI DEBUGGING ####
            moves_stringified = [str(move1) for move1 in moves]
            move = input("Enter move.")
            # Loop until valid move is selected
            while move not in moves_stringified:
                print("Enter a valid move")
                move = input("")
            move_index = moves_stringified.index(move)
            board = boards[move_index]
            move = moves[move_index]
        else:
            #### START OF HUMAN GUI ####
            #### IGNORE FOR NON-GUI DEBUGGING ####
            highlight = {}
            current_board = start_board.copy()
            step_moves = []
            move = []
            left_used = 0
            right_used = 0
            left_max = 1 + (roll[0] == roll[1])
            right_max = 1 + (roll[0] == roll[1])
            
            # Iterate through dice roll to find all legal moves for sub-turn
            for i in range(len(roll)):
                step_moves += get_legal_move(colour, current_board, roll[i])

            # Make dictionary storing start-end relationships for highlight pieces on screen
            for m in step_moves:
                if m[0] in highlight:
                    highlight[m[0]].append(m[1])
                else:
                    highlight[m[0]] = [m[1]]

            # Remove duplicates in highlight
            for start in highlight:
                highlight[start] = list(set(highlight[start]))

            starts = highlight.keys()
            start_checkers = []

            # Create highlight checkers
            for start in starts:
                if colour == 1:
                    start_checkers.append(highlight_checker(abs(current_board[start]) - 1, start, "Images/white_highlight.png", True))
                else:
                    start_checkers.append(highlight_checker(abs(current_board[start]) - 1, start, "Images/black_highlight.png", True))

            pygame.display.update()
            move_made = 0
            selected_point = None
            
            #### START OF SELECTION LOOP ####
            
            while move_made < 2 + (roll[0] == roll[1])*2 and len(step_moves) > 0:
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        
                    #### SELECTING STARTING PIECES
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        click = pygame.mouse.get_pos()
                        
                        # Checks if a movable piece has been clicked
                        for start_checker in start_checkers:
                            if start_checker.rect.collidepoint(click):
                                x, y = start_checker.rect.center
                                # Calculation for checking what point the piece is on
                                if x >= 458:
                                    point_num = (SCREEN_WIDTH - 85 - x) // 56
                                elif x <= 417:
                                    point_num = (SCREEN_WIDTH - 130 - x) // 56
                                else:
                                    point_num = int(24.5+ 0.5*colour)
                                if y <= 346:
                                    point_num = 23 - point_num
                                points = highlight[point_num]
                                
                                # Highlight the points that the pieces can be moved to
                                highlight_bottom_points(points)
                                highlight_top_points(points)
                                if any([True for i in points if i == 26 or i == 27]):
                                    highlight_home(colour)
                                selected_point = point_num

                    #### CALCULATING MOVES ####
                        
                    if event.type == pygame.KEYDOWN and selected_point is not None:
                        if event.key == pygame.K_LEFT and left_used < left_max:
                            if selected_point in highlight:
                                
                                # Makes sure the left arrow maps to the left die on the screen
                                if colour == -1:
                                    step_move = next((m for m in step_moves if m[0] == selected_point and \
                                            ((m[1] - selected_point == roll[0] and selected_point != 24) or \
                                                (selected_point == 24 and roll[0] == m[1]+1) or \
                                                    (26 in highlight[selected_point] and m[1] == 26))), None)
                                else:
                                    step_move = next((m for m in step_moves if m[0] == selected_point and \
                                            ((m[1] - selected_point == -roll[0] and selected_point != 25) or \
                                                (selected_point == 25 and roll[0] == 24 - m[1]) or \
                                                    (27 in highlight[selected_point] and m[1] == 27))), None)
                                
                                if step_move:
                                    move.append(step_move)
                                    current_board = update_board(current_board, move[-1])
                                    move_made += 1
                                    left_used +=1
                                    
                        elif event.key == pygame.K_RIGHT and right_used < right_max:
                            if selected_point in highlight:
                                
                                # Makes sure the right arrow maps to the right die on the screen
                                if colour == -1:
                                    step_move = next((m for m in step_moves if m[0] == selected_point and \
                                        ((m[1] - selected_point == roll[1] and selected_point != 24) or \
                                            (selected_point == 24 and roll[1] == m[1]+1) or \
                                                (26 in highlight[selected_point] and m[1] == 26))), None)
                                else:
                                    step_move = next((m for m in step_moves if m[0] == selected_point and \
                                        ((m[1] - selected_point == -roll[1] and selected_point != 25) or \
                                            (selected_point == 25 and roll[1] == 24 - m[1]) or \
                                                (27 in highlight[selected_point] and m[1] == 27))), None)
                                if step_move:
                                    move.append(step_move)
                                    current_board = update_board(current_board, move[-1])
                                    move_made += 1
                                    right_used += 1

                    #### END OF MOVE CALCULATIONS ####
                    
                    if event.type == pygame.MOUSEBUTTONUP and move_made < 2 + (roll[0] == roll[1])*2:
                        # Clear highlights
                        window.fill(black)
                        update_screen(background, white_score, black_score, current_board, w_score, b_score, True, score_to = score_to)
                        display_dice(colour, roll[0], roll[1])

                        # Recreate start checkers without highlights
                        start_checkers = []
                        for start in starts:
                            if colour == 1:
                                start_checkers.append(highlight_checker(abs(current_board[start]) - 1, start, "Images/white_highlight.png", True))
                            else:
                                start_checkers.append(highlight_checker(abs(current_board[start]) - 1, start, "Images/black_highlight.png", True))
                        pygame.display.update()
                        
                #### END OF MOVE SELECTION ####
                # Recalculate step moves and highlight dictionary after a move is made
                if move_made > 0:
                    step_moves = []
                    for i in range(len(roll)):
                        # Makes sure not to allow moves using left/right dice once they've been used already
                        if not ((i == 0 and left_max <= left_used) or (i == 1 and right_max <= right_used)):
                            step_moves += get_legal_move(colour, current_board, roll[i])
                    
                    # No possible moves, so end turn
                    if len(step_moves) == 0:
                        return move, current_board
                    highlight = {}
                    for m in step_moves:
                        if m[0] in highlight:
                            highlight[m[0]].append(m[1])
                        else:
                            highlight[m[0]] = [m[1]]

                    for start in highlight:
                        highlight[start] = list(set(highlight[start]))
                    if colour == 1:
                        start_checkers = [
                            highlight_checker(abs(current_board[start]) - 1, start, "Images/white_highlight.png", True)
                            for start in highlight.keys()
                        ]
                    else:
                        start_checkers = [
                            highlight_checker(abs(current_board[start]) - 1, start, "Images/black_highlight.png", True)
                            for start in highlight.keys()
                        ]
                    pygame.display.update()

            update_screen(background, white_score, black_score, current_board, w_score, b_score, True, score_to = score_to)
            pygame.display.update()
            board = current_board          
    else:
        if GUI_FLAG:
            update_screen(background, white_score, black_score, board, w_score, b_score, True, score_to = score_to)
        if commentary:
            print("No valid moves available")
    return move, board
########################
## END OF HUMAN PLAY ##
########################

def greedy_play(moves, boards, current_board, player, roll, weights=None):
    """Greedy agent makes a move

    Args:
        moves ([(int, int)]): Start-end pairs of all valid moves
        boards ([int]): Board representation
        current_board ([int]): Board before greedy plays
        player (int): The player making the move

    Returns:
        [(int, int)]: The move made
        [int]: The board resulting from move made
    """
    # Identify scores for each move's resulting board state and make sure scores[i] = boards[i]
    if weights is None:
        scores = [evaluate(current_board, boards[i], player, weights) for i in range(len(moves))]
    else:
        scores = [genetic_evaluate(current_board, boards[i], player, weights) for i in range(len(moves))]
    # Match scores, boards and moves together and sort in descending order of scores
    sorted_triplets = sorted(zip(scores, boards, moves), key=lambda x: x[0], reverse=True)
    sorted_scores, sorted_boards, sorted_moves = zip(*sorted_triplets)
    
    max_score = [i for i in sorted_scores if i==max(scores)]
    if len(max_score) > 1:
        if test:
            print("equal boards")
            for i in range(len(max_score)): print(sorted_boards[i])
        chosen_boards = tiebreak(sorted_boards[0:len(max_score)], current_board, player)
        
        chosen_inv_boards = invert_greedy(boards, current_board, player, weights, moves)
        if len(chosen_inv_boards) != len(chosen_boards):
            if test:
                print("board num mismatch", roll, player)
                print(len(chosen_boards), len(chosen_inv_boards))
                print(current_board,"\n")
                for i in chosen_boards: print(i)
                for j in chosen_inv_boards: print(j)
            raise Exception("The inverse and forward player have different boards available")
        elif len(chosen_boards) > 1:
            chosen_board = chosen_boards[randint(0, len(chosen_boards)-1)]
            chosen_inv_board = chosen_inv_boards[randint(0, len(chosen_inv_boards)-1)]
        else:
            chosen_board = chosen_boards.pop()
            chosen_inv_board = chosen_inv_boards.pop()
        
        while chosen_inv_board != chosen_board and (len(chosen_boards) > 1 and len(chosen_inv_boards) > 1):
            chosen_inv_board = chosen_inv_boards[randint(0, len(chosen_inv_boards)-1)]
            chosen_board = chosen_boards[randint(0, len(chosen_boards)-1)]
        if chosen_inv_board != chosen_board:
            if test:
                print(current_board)
                print(chosen_inv_board)
                print(chosen_board)
            raise Exception("There exists only one board per player and they're different")
        chosen_move = [sorted_moves[sorted_boards.index(chosen_board)]]
    else:
        chosen_move = [sorted_moves[0]]
        chosen_board = sorted_boards[0]
    
    return chosen_move, chosen_board, max_score[0]

def adaptive_play(moves, boards, player, turn, current_board, roll, player_score, opponent_score, cube_val, first_to, weights=None):
    if turn == 1:
        move, board, _ =  greedy_play(moves, boards, current_board, player, roll)
        move = move.pop()
        return move, board
    elif all_checkers_home(player, current_board) and all_past(current_board):
        if abs(current_board[int(26.5+(player/2))]) >= 7:
            return adaptive_race(moves, boards, player)
        else:
            # Use greedy positioning and bearing off techniques until Matussek is applicable
            move, board, _ = greedy_play(moves, boards, current_board, player, roll, weights=[10.0, 21.0, 12.0, 11.0, 15.0, 0.5664383320165035, 10.0, 4.0, 25.0, 6.0, 0.6461166029382669, 0.5378085318259279, 0.5831066576570856, 0.9552318750278183, 0.07412843879077036, 0.17550708535892934, 0.49191128795644823, 0.556755495835094])
            move = move.pop()
            return move, board
    elif all_checkers_home(-player, current_board) and abs(current_board[int(26.5+(player/2))]) == 0:#not all_checkers_home(player, current_board):
        # Gammon prevention
        return move_furthest_back(player, current_board, moves, boards)
        
    else:
        return adaptive_midgame(moves, boards, player, player_score, opponent_score, cube_val, first_to, weights, roll)

def deep_play(moves, boards, epoch=None, player=1):
    
    if len(boards) == 1:
        return moves[0], boards[0]
    if player == -1:
        boards = [invert_board(board) for board in boards]
    equities = []
    best_board = boards[0]
    encoded_best_board = torch.tensor(convert_board(best_board), dtype=torch.float32).unsqueeze(0)
    best_move = moves[0]
    for i in range(1, len(boards)):
        left_board = best_board
        right_board = boards[i]
        encoded_right_board = torch.tensor(convert_board(right_board), dtype=torch.float32).unsqueeze(0)
        prediction = predict(left_board, right_board, encoded_best_board, encoded_right_board, epoch)
        if prediction < 0.5:
            best_board = right_board
            best_move = moves[i]
            encoded_best_board = torch.tensor(convert_board(best_board), dtype=torch.float32).unsqueeze(0)
        # print(equities)
    # print(equities)
    if player == -1:
        best_board = invert_board(best_board)
    return best_move, best_board

###############
## MAIN BODY ##
###############
def backgammon(score_to=1,whitestrat="GREEDY", whiteweights = None, blackstrat="GREEDY", blackweights = None, double_point=None, double_drop=None, starting_board=None, w_start_score = 0, b_start_score = 0):
    """Play the backgammon game

    Args:
        score_to (int, optional): What score reached before terminate. Defaults to 1.
        whitestrat (str, optional): Player 1 alg. Defaults to "GREEDY".
        weights1 (list(float), optional): Weights for genetic player 1. Defaults to None.
        blackstrat (str, optional): Player 2 alg. Defaults to "GREEDY".
        weights2 (list(float), optional): Weights for genetic player 2. Defaults to None.

    Returns:
        list(int), int, list(int), int: Win vector for white player
                                        Score for white player
                                        Win vector for black player
                                        Score for black player
    """
    #### SCORE INITIALISATION ####
    w_score, b_score = int(w_start_score),int(b_start_score)
    prev_score = [0,0]
    p1vector = [0,0,0] 
    pminus1vector = [0,0,0] 
    game = 1
    if commentary: print(whitestrat, blackstrat)
    #### MAIN LOOP ####
    while max([w_score, b_score]) < score_to:
        black_equity = []
        white_equity = []
        # starting_board = generate_random_race_board()
        if starting_board is None:
            board = make_board()
        else:
            board = starting_board
            # print(board)
            
        print_board(board)
        white_boards = []
        black_boards = []
        if GUI_FLAG:
            update_screen(background, white_score, black_score, board, w_score, b_score, True, score_to = score_to)
        time_step = 1
        cube_val = 1
        double_player = 0
        #### GAME LOOP ####
        while not game_over(board) and not is_error(board):
            if GUI_FLAG:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
            #### FIRST TURN ####     
            if time_step == 1 and game == 1:
                # Each player rolls a die to determine who moves first
                white_roll, black_roll = roll_dice()
                #### DISPLAY FIRST DICE ROLL FOR WHO GOES FIRST ####
                if GUI_FLAG:
                    
                    pygame.display.update()
                    framesPerSec.tick(30)
                    update_screen(background, white_score, black_score, board, w_score, b_score, True, score_to = score_to)
                    pygame.display.update()
                    sleep(1)
                    # Loop 60 times for rolling animation
                    for i in range(60):
                        white_roll, black_roll = roll_dice()
                        window.blit(black_dice[black_roll-1], (SCREEN_WIDTH//4-28, SCREEN_HEIGHT//2))
                        window.blit(white_dice[white_roll-1], (3*SCREEN_WIDTH//4+28, SCREEN_HEIGHT//2))
                        pygame.display.update()
                
                # In case of draw, keep rolling until one roll is larger than the other
                while black_roll == white_roll:
                    white_roll, black_roll = roll_dice()
                    if GUI_FLAG:
                        window.blit(black_dice[black_roll-1], (SCREEN_WIDTH//4-28, SCREEN_HEIGHT//2))
                        window.blit(white_dice[white_roll-1], (3*SCREEN_WIDTH//4+28, SCREEN_HEIGHT//2))
                        pygame.display.update()
                if USER_PLAY or GUI_FLAG:
                    sleep(1)
                if GUI_FLAG:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            
                #### END OF PLAYER 1 SELECTION ####
                      
                if commentary:
                    print(f"Black rolled {black_roll}")
                    print(f"White rolled {white_roll}")
                # Black starts first
                if black_roll > white_roll:
                    player1 = -1
                    player1strat = blackstrat
                    weights1 = blackweights
                    player1score = b_score
                    player2 = 1
                    player2strat = whitestrat
                    weights2 = whiteweights
                    player2score = w_score
                    black_boards.append(board)
                    white_boards.append(board)
                    if GUI_FLAG:
                        background.render()
                        window.blit(black_dice[black_roll-1], (SCREEN_WIDTH//4-28, SCREEN_HEIGHT//2))
                        window.blit(black_dice[white_roll-1], (SCREEN_WIDTH//4+28, SCREEN_HEIGHT//2))
                else:
                    # White starts first
                    player1 = 1
                    player1strat = whitestrat
                    weights1 = whiteweights
                    player1score = w_score
                    player2 = -1
                    player2strat = blackstrat
                    weights2 = blackweights
                    player2score = b_score
                    white_boards.append(board)
                    black_boards.append(board)
                    if GUI_FLAG:
                        background.render()
                        window.blit(white_dice[black_roll-1], (3*SCREEN_WIDTH//4-28, SCREEN_HEIGHT//2))
                        window.blit(white_dice[white_roll-1], (3*SCREEN_WIDTH//4+28, SCREEN_HEIGHT//2))
                if test: first_turn(player1)
                if GUI_FLAG:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                    update_screen(background, white_score, black_score, board, w_score, b_score, score_to = score_to)
                    pygame.display.update()
                    sleep(1)
                # Initial roll made up of both starting dice
                roll = [black_roll, white_roll]
                moves1, boards1 = get_valid_moves(player1, board, roll)
                if test:
                    check_moves(board, boards1, player1, roll)
                print_board(board)
                if commentary:
                    print(f"Player {player1} rolled {roll}")
            else:
                # All other rolls are generated on spot
                if time_step > 1:
                    has_double_rejected = False   
                    equity = calc_equity(board, player1)
                    # write_equity(equity, "BasicEquity")
                    if can_double(double_player, player1, w_score, b_score, score_to, prev_score):
                        cube_val, double_player, has_double_rejected= double_process(player1strat, player1, board, player2strat, cube_val, double_player, player1score, player2score, score_to, double_point, double_drop)
                    if has_double_rejected:
                        if commentary: print("Double Rejected")
                        board = get_double_rejected_board(player1)
                        break
                    elif commentary:
                        print("Double accepted")
                        print(f"Cube now {cube_val}")
                    elif GUI_FLAG:
                        update_screen(background, white_score, black_score, board, w_score, b_score, True, score_to = score_to)
                moves1, boards1, roll = start_turn(player1, board)
            if test:
                save_roll(roll, player1)
            #### END OF FIRST TURN ####
            
            if USER_PLAY or GUI_FLAG:
                sleep(0.5)
            if GUI_FLAG:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
            #### WHITE IS PLAYER 1 ####
            if player1 == 1:
                #### WHITE PLAYER 1'S TURN ####
                if len(moves1) > 0:
                    if player1strat == "USER":
                        move, board = human_play(moves1, boards1, board, roll, player1)
                    elif player1strat == "RANDOM":
                        board, move = randobot_play(roll, moves1, boards1)
                    elif player1strat == "GREEDY":
                        move, board, evaluation = greedy_play(moves1, boards1, board, player1, roll)
                        if test:
                            equity = calc_advanced_equity(board, player1, player1score, player2score, cube_val, score_to)
                            compare_eval_equity(evaluation, equity)
                        move = move.pop()
                        
                    elif player1strat == "GENETIC":
                        move, board, evaluation = greedy_play(moves1, boards1, board, player1, roll, weights1)
                        if test:
                            equity = calc_advanced_equity(board, player1, player1score, player2score, cube_val, score_to)
                            compare_eval_equity(evaluation, equity)
                        move = move.pop()
                    elif player1strat == "EXPECTIMAX":
                        if all_past(board):
                            move, board = greedy_play(moves1, boards1, board, player1, roll)
                        else:
                            move, board = expectimax_play(moves1, boards1, player1)
                        move = move.pop()
                    elif player1strat == "ADAPTIVE":
                        white_equity.append(calc_advanced_equity(board, player1, player1score, player2score, cube_val, score_to, weights1))
                        move, board = adaptive_play(moves1, boards1, player1, time_step, board, roll, player1score, player2score, cube_val, score_to, weights1)
                    elif player1strat == "DEEP":
                        move, board = deep_play(moves1, boards1, weights1)
                    white_boards.append(board)
                    # write_move_equities(board, roll, player1) 
                    if commentary:
                        print(f"Move Taken: {move}")
                    if GUI_FLAG:
                        for event in pygame.event.get():
                            if event.type == QUIT:
                                pygame.quit()
                        if len(moves1) > 0 and player1strat != "USER":
                            start_point, start_checkers, end_point, end_checkers = parse_move(board, move)
                            for i in range(len(start_point)):
                                highlight_checker(start_checkers[i], start_point[i], "Images/white_highlight.png")
                                pygame.display.update()
                                sleep(1)
                                
                                highlight_checker(end_checkers[i], end_point[i], "Images/white_highlight.png")
                                pygame.display.update()
                                sleep(1)
                                
                                
                                highlight_checker(end_checkers[i], end_point[i], "Images/white_pawn.png")
                                pygame.display.update()
                        update_screen(background, white_score, black_score, board, w_score, b_score, True, score_to = score_to)
                        pygame.display.update()
                        sleep(1)
                    #### END OF WHITE PLAYER 1'S TURN ####
                else:
                    if commentary:
                        print("No move can be played")
                print_board(board)
                
                # Game ends?
                if is_error(board):
                    print("Error detected")
                    exit()
                if game_over(board):
                    break
                if USER_PLAY or GUI_FLAG:
                    sleep(1)
                
                #### BLACK PLAYER 2'S TURN ####
                has_double_rejected = False
                equity = calc_equity(board, player2)
                # write_equity(equity, "BasicEquity")  
                if can_double(double_player, player2, w_score, b_score, score_to, prev_score):
                    cube_val, double_player, has_double_rejected= double_process(player2strat, player2, board, player1strat, cube_val, double_player, player2score, player1score, score_to, double_point, double_drop)
                if has_double_rejected:
                    if commentary: print("Double Rejected")
                    board = get_double_rejected_board(player2)
                    break
                elif commentary:
                    print("Double accepted")
                    print(f"Cube now {cube_val}")
                elif GUI_FLAG:
                    update_screen(background, white_score, black_score, board, w_score, b_score, True, score_to = score_to)
                moves2, boards2, roll = start_turn(player2, board)
                
                if test:
                    save_roll(roll, player2)
                if len(moves2) > 0:
                    if player2strat == "USER":
                        move, board = human_play(moves2, boards2, board, roll, player2)
                    elif player2strat == "RANDOM":
                        board, move = randobot_play(roll, moves2, boards2)
                    elif player2strat == "GREEDY":
                        move, board, evaluation = greedy_play(moves2, boards2, board, player2, roll)
                        if test:
                            equity = calc_advanced_equity(board, player2, player2score, player1score, cube_val, score_to)
                            compare_eval_equity(evaluation, equity)
                        move = move.pop()
                    elif player2strat == "GENETIC":
                        move, board, evaluation = greedy_play(moves2, boards2, board, player2, roll, weights2)
                        if test:
                            equity = calc_advanced_equity(board, player2, player2score, player1score, cube_val, score_to)
                            compare_eval_equity(evaluation, equity)
                        move = move.pop()
                    elif player2strat == "EXPECTIMAX":
                        if all_past(board):
                            move, board = greedy_play(moves2, boards2, board, player2, roll)
                        else:
                            move, board = expectimax_play(moves2, boards2, player2)
                        move = move.pop()
                    elif player2strat == "ADAPTIVE":
                        black_equity.append(calc_advanced_equity(board, player2, player2score, player1score, cube_val, score_to, weights2))
                        move, board = adaptive_play(moves2, boards2, player2, time_step, board, roll, player2score, player1score, cube_val, score_to, weights2)
                    elif player2strat == "DEEP":
                        move, board = deep_play(moves2, boards2, weights2, -1)
                    black_boards.append(board)    
                    # write_move_equities(board, roll, player2)
                    if commentary:
                        print(f"Move Taken: {move}")
                else:
                    if commentary:
                        print("No move can be played")
                
                if GUI_FLAG:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                    if len(moves2) > 0 and player2strat != "USER":
                        start_point, start_checkers, end_point, end_checkers = parse_move(board, move)
                        for i in range(len(start_point)):
                            highlight_checker(start_checkers[i], start_point[i], "Images/black_highlight.png")
                            pygame.display.update()
                            sleep(1)
                            
                            highlight_checker(end_checkers[i], end_point[i], "Images/black_highlight.png")
                            pygame.display.update()
                            sleep(1)
                            
                            
                            highlight_checker(end_checkers[i], end_point[i], "Images/black_pawn.png")
                            pygame.display.update()
                    update_screen(background, white_score, black_score, board, w_score, b_score, True, score_to = score_to)
                    pygame.display.update()
                    sleep(1)
                    
                #### END OF BLACK PLAYER 2'S TURN ####
            else:
                
                #### BLACK PLAYER 1'S TURN ####
                
                if len(moves1) > 0:
                    if player1strat == "USER":
                        move, board = human_play(moves1, boards1, board, roll, player1)
                    elif player1strat == "RANDOM":
                        board, move = randobot_play(roll, moves1, boards1)
                    elif player1strat == "GREEDY":
                        move, board, evaluation = greedy_play(moves1, boards1, board, player1, roll)
                        if test:
                            equity = calc_advanced_equity(board, player1, player1score, player2score, cube_val, score_to)
                            compare_eval_equity(evaluation, equity)
                        move = move.pop()
                    elif player1strat == "GENETIC":
                        move, board, evaluation = greedy_play(moves1, boards1, board, player1, roll, weights1)
                        if test:
                            equity = calc_advanced_equity(board, player1, player1score, player2score, cube_val, score_to)
                            compare_eval_equity(evaluation, equity)
                        move = move.pop()
                    elif player1strat == "EXPECTIMAX":
                        if all_past(board):
                            move, board = greedy_play(moves1, boards1, board, player1, roll)
                        else:
                            move, board = expectimax_play(moves1, boards1, player1)
                        move = move.pop()
                    elif player1strat == "ADAPTIVE":
                        black_equity.append(calc_advanced_equity(board, player1, player1score, player2score, cube_val, score_to, weights1))
                        move, board = adaptive_play(moves1, boards1, player1, time_step, board, roll, player1score, player2score, cube_val, score_to, weights1)
                    elif player1strat == "DEEP":
                        move, board = deep_play(moves1, boards1, weights1, -1)
                    black_boards.append(board)
                    # write_move_equities(board, roll, player1)
                    if commentary:    
                        print(f"Move Taken: {move}")
                else:
                    if commentary:
                        print("No move can be played")
                        
                if GUI_FLAG:
                    for event in pygame.event.get():
                            if event.type == QUIT:
                                pygame.quit()
                    if len(moves1) > 0 and player1strat != "USER":          
                        start_point, start_checkers, end_point, end_checkers = parse_move(board, move)
                        for i in range(len(start_point)):
                            highlight_checker(start_checkers[i], start_point[i], "Images/black_highlight.png")
                            pygame.display.update()
                            sleep(1)
                            
                            highlight_checker(end_checkers[i], end_point[i], "Images/black_highlight.png")
                            pygame.display.update()
                            sleep(1)
                            
                            
                            highlight_checker(end_checkers[i], end_point[i], "Images/black_pawn.png")
                            pygame.display.update()
                            
                    update_screen(background, white_score, black_score, board, w_score, b_score, True, score_to = score_to)
                    pygame.display.update()
                    sleep(1)
                    
                #### END OF BLACK PLAYER 1'S TURN ####
                
                if commentary:
                    print_board(board)
                if USER_PLAY or GUI_FLAG:
                    sleep(1)
                if is_error(board):
                    exit()
                if game_over(board):
                    break
                
                #### WHITE PLAYER 2'S TURN ####
                has_double_rejected = False   
                equity = calc_equity(board, player2)
                # write_equity(equity, "BasicEquity")
                if can_double(double_player, player2, w_score, b_score, score_to, prev_score):
                    cube_val, double_player, has_double_rejected= double_process(player2strat, player2, board, player1strat, cube_val, double_player, player2score, player1score, score_to, double_point, double_drop)
                if has_double_rejected:
                    if commentary: print("Double Rejected")
                    board = get_double_rejected_board(player2)
                    break
                elif commentary:
                    print("Double accepted")
                    print(f"Cube now {cube_val}")
                elif GUI_FLAG:
                    update_screen(background, white_score, black_score, board, w_score, b_score, True, score_to = score_to)
                moves2, boards2, roll = start_turn(player2, board)
                if test:
                    save_roll(roll, player2)
                if len(moves2) > 0:
                    if player2strat == "USER":
                        move, board = human_play(moves2, boards2, board, roll, player2)
                    elif player2strat == "RANDOM":
                        board, move = randobot_play(roll, moves2, boards2)
                    elif player2strat == "GREEDY":
                        move, board, evaluation = greedy_play(moves2, boards2, board, player2, roll)
                        if test:
                            equity = calc_advanced_equity(board, player2, player2score, player1score, cube_val, score_to)
                            compare_eval_equity(evaluation, equity)
                        move = move.pop()
                    elif player2strat == "GENETIC":
                        move, board, evaluation = greedy_play(moves2, boards2, board, player2, roll, weights2)
                        if test:
                            equity = calc_advanced_equity(board, player1, player1score, player2score, cube_val, score_to)
                            compare_eval_equity(evaluation, equity)
                        move = move.pop()
                    elif player2strat == "EXPECTIMAX":
                        if all_past(board):
                            move, board = greedy_play(moves2, boards2, board, player2, roll)
                        else:
                            move, board = expectimax_play(moves2, boards2, player2)
                        move = move.pop()
                    elif player2strat == "ADAPTIVE":
                        white_equity.append(calc_advanced_equity(board, player2, player2score, player1score, cube_val, score_to, weights2))
                        move, board = adaptive_play(moves2, boards2, player2, time_step, board, roll, player2score, player1score, cube_val, score_to, weights2)
                    elif player2strat == "DEEP":
                        move, board = deep_play(moves2, boards2, weights2)
                    white_boards.append(board)    
                    # write_move_equities(board, roll, player2)
                    if commentary:
                        print(f"Move Taken: {move}")
                else:
                    if commentary:
                        print("No move can be played")
                if GUI_FLAG:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                    if len(moves2) > 0 and player2strat != "USER":
                        start_point, start_checkers, end_point, end_checkers = parse_move(board, move)
                        
                    
                        for i in range(len(start_point)):
                            highlight_checker(start_checkers[i], start_point[i], "Images/white_highlight.png")
                            pygame.display.update()
                            sleep(1)
                            
                            highlight_checker(end_checkers[i], end_point[i], "Images/white_highlight.png")
                            pygame.display.update()
                            sleep(1)
                            
                            
                            highlight_checker(end_checkers[i], end_point[i], "Images/white_pawn.png")
                            pygame.display.update()
                    update_screen(background, white_score, black_score, board, w_score, b_score, True, score_to = score_to)
                    pygame.display.update()
                    sleep(1)
                
            #### END OF WHITE PLAYER 2'S TURN ####
            
            print_board(board)
            if is_error(board):
                exit()
            time_step +=1
            if USER_PLAY:
                sleep(1)
        #### CHECKS FOR GAME OVER AND WINNING POINTS ####
        if game_over(board):
            crawford_game = is_crawford_game(w_score, b_score, score_to, prev_score)
            prev_score = [w_score, b_score]
            game += 1
            # timestep(time_step)
            if commentary:
                print("GAME OVER")
                
            if is_backgammon(board):
                if board[26] == -15:
                    # write_board_points([invert_board(i) for i in black_boards], 3)
                    # write_board_points(white_boards, 0)
                    pminus1vector[2] +=cube_val
                    if commentary:
                        print("Player -1 win")
                else:
                    # write_board_points(white_boards, 3)
                    # write_board_points([invert_board(i) for i in black_boards], 0)
                    p1vector[2] +=cube_val
                    if commentary:
                        print("Player 1 win")
                if commentary:
                    print("By backgammon")
                    
            elif is_gammon(board):
                if board[26] == -15:
                    # write_board_points([invert_board(i) for i in black_boards], 2)
                    # write_board_points(white_boards, 0)
                    pminus1vector[1] +=cube_val
                    if commentary:
                        print("Player -1 win")
                else:
                    # write_board_points(white_boards, 2)
                    # write_board_points([invert_board(i) for i in black_boards], 0)
                    p1vector[1] +=cube_val
                    if commentary:
                        print("Player 1 win")
                if commentary:
                    print("By gammon")
                    
            else:
                if board[26] == -15:
                    # write_board_points([invert_board(i) for i in black_boards], 1)
                    # write_board_points(white_boards, 0)
                    pminus1vector[0] +=cube_val
                    if commentary:
                        print("Player -1 win")
                else:
                    # write_board_points(white_boards, 1)
                    # write_board_points([invert_board(i) for i in black_boards], 0)
                    p1vector[0] +=cube_val
                    if commentary:
                        print("Player 1 win")
            
            # print(w_score, b_score)
            if board[26] == -15:
                # Black won, so it moves first next round
                player1 = -1
                player1strat = blackstrat
                weights1 = blackweights
                player2 = 1
                player2strat = whitestrat
                weights2 = whiteweights
                b_score = pminus1vector[0] + 2*pminus1vector[1] + 3*pminus1vector[2]
                player1score = b_score
                player2score = w_score
                # for eq in black_equity:
                #     write_equity(eq, "WinnerEquity")
                # for eq in white_equity:
                #     write_equity(eq, "LoserEquity")
                
            else:
                # White won, so it moves first next round
                player1 = 1
                player1strat = whitestrat
                weights1 = whiteweights
                player2 = -1
                player2strat = blackstrat
                weights2 = blackweights
                w_score = p1vector[0] + 2*p1vector[1] + 3*p1vector[2]
                player1score = w_score
                player2score = b_score
                # for eq in white_equity:
                #     write_equity(eq, "WinnerEquity")
                # for eq in black_equity:
                #     write_equity(eq, "LoserEquity")
            cube_val = 1
            time_step = 1
            if commentary: print(f"White: {w_score} Black: {b_score}")
            if GUI_FLAG:
                update_screen(background, white_score, black_score, board, w_score, b_score, True, score_to = score_to)    
        #### CHECKS FOR GAME OVER AND WINNING POINTS ####
            
            print(w_score, b_score)
    return p1vector, w_score, pminus1vector, b_score


def collect_data(p1strat, pminus1strat, first_to):
    myFile = "./Data/cubefuldeep4.5vdeep1.txt"
    white_tot, black_tot = 0,0
    white_wins, black_wins = 0,0
    first_to = 25
    adaptive_weights = [0.9966066885314592, -0.9916984096898946, 0.3106830724424913, 0.529168163359478, -0.4710732676896102, 0.5969523488654117, 0.36822981983332415, 0.38958074063216697, 0.02676397245530815, 0.08588282381449319, 0.06094873757931751, 1.1095422351658368, 0.47764793610307643, 0.040753486445243126, 0.5495226441839489, 0.8875009606764003, 0.9333344067224983, 0.1340269726805713, 0.1978868967026618, 1.2096547126804458, 2.379707426788366, 0.6465298771549699, 0.509196585225148, 0.261875669397977, 0.36883752029556166, -0.481342015629518, 0.7098436807557322, 1.0250219115287624, 0.5739284594183071, 0.1796876959733017, 0.2679991261065485]
    genetic_weights = [10.0, 21.0, 12.0, 11.0, 15.0, 0.5664383320165035, 10.0, 4.0, 25.0, 6.0, 0.6461166029382669, 0.5378085318259279, 0.5831066576570856, 0.9552318750278183, 0.07412843879077036, 0.17550708535892934, 0.49191128795644823, 0.556755495835094]
    double_point, double_drop = 1.4325859937671366, -1.8523842372779313
    for i in range(1000):
        dataFile = open(myFile, 'a')
        p1vector,w_score,pminus1vector,b_score= backgammon(first_to, "DEEP",None, "DEEP","v1")
        dataFile.write(f"{w_score}, {b_score}\n")
        print(p1vector,w_score,pminus1vector,b_score)
        dataFile.close()
        white_tot+=w_score
        black_tot+=b_score
        if b_score >= first_to:
            black_wins += 1
        if w_score >= first_to:
            white_wins +=1
        if i % 2 == 1:
            print("score")
            print(white_tot, black_tot)
            print("Wins")
            print(white_wins, black_wins)

    # print(calc_av_eval())
    print(white_tot, black_tot)
    print(white_wins, black_wins)
    
    

           
if __name__ == "__main__":
    if len(sys.argv[:]) > 1:
        if sys.argv[1].lower() == "data":
            
            if len(sys.argv[:]) >= 3:
                print("Player 1", sys.argv[2], "Player -1", sys.argv[3])
                collect_data(sys.argv[2], sys.argv[3], 25)
            else:
                collect_data("DEEP",'ADAPTIVE',25)
            # print(calc_first())
        else:
            score_to = 25
            player1strat = sys.argv[1]
            playerminus1strat = sys.argv[2]
            if len(sys.argv) == 5:
                w_start_score = sys.argv[3]
                b_start_score = sys.argv[4]
            elif len(sys.argv) == 4 or len(sys.argv) > 5:
                print("You've done something wrong")
                print("Your command should be written in the format:")
                print("python Code/main.py USER {AI} {WhiteScore} {BlackScore}")
                print("Where {AI} is the AI you want to play")
                print("And {WhiteScore} and {BlackScore} are the scores of your incomplete match (if one exists)")
                print("If you don't have an incomplete match, don't worry putting in WhiteScore or BlackScore values")
                exit()
            weights1, weights2 = None, None
            if player1strat == "GENETIC":
            # Optimal Weights for first-to-25 victory
                weights1 = [10.0, 21.0, 12.0, 11.0, 15.0, 0.5664383320165035, 10.0, 4.0, 25.0, 6.0, 0.6461166029382669, 0.5378085318259279, 0.5831066576570856, 0.9552318750278183, 0.07412843879077036, 0.17550708535892934, 0.49191128795644823, 0.556755495835094]
            elif player1strat == "ADAPTIVE":
                weights1 = [0.9966066885314592, -0.9916984096898946, 0.3106830724424913, 0.529168163359478, -0.4710732676896102, 0.5969523488654117, 0.36822981983332415, 0.38958074063216697, 0.02676397245530815, 0.08588282381449319, 0.06094873757931751, 1.1095422351658368, 0.47764793610307643, 0.040753486445243126, 0.5495226441839489, 0.8875009606764003, 0.9333344067224983, 0.1340269726805713, 0.1978868967026618, 1.2096547126804458, 2.379707426788366, 0.6465298771549699, 0.509196585225148, 0.261875669397977, 0.36883752029556166, -0.481342015629518, 0.7098436807557322, 1.0250219115287624, 0.5739284594183071, 0.1796876959733017, 0.2679991261065485]
            if playerminus1strat == "GENETIC":
                weights2 = [10.0, 21.0, 12.0, 11.0, 15.0, 0.5664383320165035, 10.0, 4.0, 25.0, 6.0, 0.6461166029382669, 0.5378085318259279, 0.5831066576570856, 0.9552318750278183, 0.07412843879077036, 0.17550708535892934, 0.49191128795644823, 0.556755495835094]
            elif playerminus1strat == "ADAPTIVE":
                weights2 = [0.9966066885314592, -0.9916984096898946, 0.3106830724424913, 0.529168163359478, -0.4710732676896102, 0.5969523488654117, 0.36822981983332415, 0.38958074063216697, 0.02676397245530815, 0.08588282381449319, 0.06094873757931751, 1.1095422351658368, 0.47764793610307643, 0.040753486445243126, 0.5495226441839489, 0.8875009606764003, 0.9333344067224983, 0.1340269726805713, 0.1978868967026618, 1.2096547126804458, 2.379707426788366, 0.6465298771549699, 0.509196585225148, 0.261875669397977, 0.36883752029556166, -0.481342015629518, 0.7098436807557322, 1.0250219115287624, 0.5739284594183071, 0.1796876959733017, 0.2679991261065485]
            
            p1vector, w_score, pminus1vector, b_score = backgammon(score_to, player1strat, None, playerminus1strat, weights2, w_start_score= w_start_score, b_start_score=b_start_score)
            print(w_score, b_score)
    else:
        # print(calc_av_eval())
        
        # print(calc_first())
        # print(b:=update_board(make_board(),(12, 9)))
        # print(update_board(b, (9, 7)))
        score_to = 25
        player1strat = "USER"
        playerminus1strat = "DEEP"
        print(player1strat, playerminus1strat)
        weights1, weights2 = None, None
        if player1strat == "GENETIC":
            # Optimal Weights for first-to-25 victory
            weights1 = [10.0, 21.0, 12.0, 11.0, 15.0, 0.5664383320165035, 10.0, 4.0, 25.0, 6.0, 0.6461166029382669, 0.5378085318259279, 0.5831066576570856, 0.9552318750278183, 0.07412843879077036, 0.17550708535892934, 0.49191128795644823, 0.556755495835094]
        elif player1strat == "ADAPTIVE":
            weights1 = [0.9966066885314592, -0.9916984096898946, 0.3106830724424913, 0.529168163359478, -0.4710732676896102, 0.5969523488654117, 0.36822981983332415, 0.38958074063216697, 0.02676397245530815, 0.08588282381449319, 0.06094873757931751, 1.1095422351658368, 0.47764793610307643, 0.040753486445243126, 0.5495226441839489, 0.8875009606764003, 0.9333344067224983, 0.1340269726805713, 0.1978868967026618, 1.2096547126804458, 2.379707426788366, 0.6465298771549699, 0.509196585225148, 0.261875669397977, 0.36883752029556166, -0.481342015629518, 0.7098436807557322, 1.0250219115287624, 0.5739284594183071, 0.1796876959733017, 0.2679991261065485]
        if playerminus1strat == "GENETIC":
            weights2 = [10.0, 21.0, 12.0, 11.0, 15.0, 0.5664383320165035, 10.0, 4.0, 25.0, 6.0, 0.6461166029382669, 0.5378085318259279, 0.5831066576570856, 0.9552318750278183, 0.07412843879077036, 0.17550708535892934, 0.49191128795644823, 0.556755495835094]
        elif playerminus1strat == "ADAPTIVE":
            weights2 = [0.9966066885314592, -0.9916984096898946, 0.3106830724424913, 0.529168163359478, -0.4710732676896102, 0.5969523488654117, 0.36822981983332415, 0.38958074063216697, 0.02676397245530815, 0.08588282381449319, 0.06094873757931751, 1.1095422351658368, 0.47764793610307643, 0.040753486445243126, 0.5495226441839489, 0.8875009606764003, 0.9333344067224983, 0.1340269726805713, 0.1978868967026618, 1.2096547126804458, 2.379707426788366, 0.6465298771549699, 0.509196585225148, 0.261875669397977, 0.36883752029556166, -0.481342015629518, 0.7098436807557322, 1.0250219115287624, 0.5739284594183071, 0.1796876959733017, 0.2679991261065485]
        # start=time()
        p1vector, w_score, pminus1vector, b_score = backgammon(score_to,player1strat,weights1,playerminus1strat,weights2)
        # print(time()-start)
        print(p1vector,pminus1vector)
        
        

# [0.9966066885314592, -0.9916984096898946, 0.3106830724424913, 0.529168163359478, -0.4710732676896102, 0.5969523488654117, 0.36822981983332415, 0.38958074063216697, 0.02676397245530815, 0.08588282381449319, 0.06094873757931751, 1.1095422351658368, 0.47764793610307643, 0.040753486445243126, 0.5495226441839489, 0.8875009606764003, 0.9333344067224983, 0.1340269726805713, 0.1978868967026618, 1.2096547126804458, 2.379707426788366, 0.6465298771549699, 0.509196585225148, 0.261875669397977, 0.36883752029556166, -0.481342015629518, 0.7098436807557322, 1.0250219115287624, 0.5739284594183071, 0.1796876959733017, 0.2679991261065485]
# [0.12956605564741974, -0.7036928210496392, 0.4074025237866028, 0.4799756327265836, -0.40291585703358956, 0.3886114104807685, 0.8366009902042647, 0.9238223233657615, 0.012446338390701084, 0.6878007594142832, 0.49076763383618993, 2.5772096603066896, 0.01872547641265143, 0.3567620173061472, 0.0054600651603922135, 0.7247670802779068, 0.6432669390280996, 0.2816665181247421, 0.9576215830934188, 2.1132790049996677, 1.018767811919806, 0.48013492994273, 0.371329501821977, 0.988761590907909, 0.990675955445724, -0.8410175300701395, 0.404656923138184, 0.5106857301341181, 0.36374552600495513, 0.9250343710417397, 0.5328884028513421]
# [0.060036560799419325, -0.6322214931321674, 0.5777022031876009, 0.19049173665823282, -0.9935437697548243, 0.42447257975140584, 0.2505055135534632, 0.2962012595086625, 0.5176038621467028, 0.8488459391393087, 0.7859466567026897, 2.910243554132832, 0.9341629051769885, 0.6773999252120184, 0.054751057448326645, 0.5242456934735903, 0.9943016519994361, 0.18848883560844898, 0.31265078759496456, 2.9879011731714895, 1.5979572440850984, 0.20074714964701634, 0.05174938532749007, 0.5392755915824913, 0.6016371387913554, -0.8585670279590657, 0.1309780203636698, 0.3925562540338223, 0.838342613015815, 0.403251355165662, 0.03029734913362181]
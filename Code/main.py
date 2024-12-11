import pygame
import sys
import numpy as np
from pygame.locals import *
from random_agent import *
from turn import *
from time import sleep 
from greedy_agent import *
from constants import *
from gui import *
from genetic_agent import genetic
from data import *

global background
global white_score
global black_score
global w_score 
global b_score

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
    return moves, boards, roll

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
            highlight = {}
            current_board = start_board.copy()
            step_moves = []
            move = []
            left_used = 0
            right_used = 0
            left_max = 1 + (roll[0] == roll[1])
            right_max = 1 + (roll[0] == roll[1])
            # Generate legal moves
            for i in range(len(roll)):
                step_moves += get_legal_move(colour, current_board, roll[i])

            # Populate the highlight dictionary
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

            while move_made < 2 + (roll[0] == roll[1])*2:
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        click = pygame.mouse.get_pos()
                        for start_checker in start_checkers:
                            if start_checker.rect.collidepoint(click):
                                x, y = start_checker.rect.center
                                if x >= 458:
                                    point_num = (SCREEN_WIDTH - 85 - x) // 56
                                elif x <= 417:
                                    point_num = (SCREEN_WIDTH - 141 - x) // 56
                                else:
                                    point_num = int(24.5+ 0.5*colour)
                                if y <= 346:
                                    point_num = 23 - point_num
                                print(point_num)
                                points = highlight[point_num]
                                highlight_bottom_points(points)
                                highlight_top_points(points)
                                selected_point = point_num

                    if event.type == pygame.KEYDOWN and selected_point is not None:
                        if event.key == pygame.K_LEFT and left_used < left_max:
                            if highlight[selected_point]:
                                if colour == -1:
                                    step_move = next((m for m in step_moves if m[0] == selected_point and (m[1] - selected_point == roll[0]) or (selected_point == 24 and roll[0] == m[1]+1)), None)
                                else:
                                    step_move = next((m for m in step_moves if m[0] == selected_point and (m[1] - selected_point == -roll[0]) or (selected_point == 25 and roll[0] == 24-m[1])), None)
                                print(step_move)
                                if step_move:
                                    move.append(step_move)
                                    current_board = update_board(current_board, move[-1])
                                    move_made += 1
                                    left_used +=1
                                    
                        elif event.key == pygame.K_RIGHT and right_used < right_max:
                            if colour == -1:
                                step_move = next((m for m in step_moves if m[0] == selected_point and (m[1] - selected_point == roll[1]) or (selected_point == 24 and roll[1] == m[1]+1)), None)
                            else:
                                step_move = next((m for m in step_moves if m[0] == selected_point and (m[1] - selected_point == -roll[1]) or (selected_point == 25 and roll[1] == 24-m[1])), None)
                            if step_move:
                                move.append(step_move)
                                current_board = update_board(current_board, move[-1])
                                move_made += 1
                                right_used += 1
                            
                    if event.type == pygame.MOUSEBUTTONUP and move_made < 2 + (roll[0] == roll[1])*2:
                        # Clear highlights
                        window.fill(black)
                        update_screen(background, white_score, black_score, current_board, w_score, b_score, True)
                        display_dice(colour, roll[0], roll[1])

                        # Recreate start checkers without highlights
                        start_checkers = []
                        for start in starts:
                            if colour == 1:
                                start_checkers.append(highlight_checker(abs(current_board[start]) - 1, start, "Images/white_highlight.png", True))
                            else:
                                start_checkers.append(highlight_checker(abs(current_board[start]) - 1, start, "Images/black_highlight.png", True))
                        pygame.display.update()

                # Recalculate step moves and highlight dictionary after a move is made
                if move_made > 0:
                    step_moves = []
                    for i in range(len(roll)):
                        step_moves += get_legal_move(colour, current_board, roll[i])

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

            update_screen(background, white_score, black_score, current_board, w_score, b_score, True)
            pygame.display.update()
            board = current_board          
    else:
        print("No valid moves available")
        
    return move, board


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

def greedy_play(moves, boards, current_board, player, roll):
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
    scores = [evaluate(current_board, boards[i], player) for i in range(len(moves))]
    sorted_triplets = sorted(zip(scores, boards, moves), key=lambda x: x[0], reverse=True)
    sorted_scores, sorted_boards, sorted_moves = zip(*sorted_triplets)
    log(current_board, roll, sorted_moves[0], list(sorted_boards)[0])
    return [sorted_moves[0]], list(sorted_boards)[0]
    
    
    
def genetic_play(moves, boards, weights):
    pass
    # Weights are assigned to different types of moves
    # Examine boards and compare which boards should be in which categories
    # Use weights to determine which category is being used, roulette wheel
    # For tiebreakers, check if moves fall into multiple categories. 
    # Evaluate weights assigned to the other categories, perform roulette wheel again
    # If absolute tiebreak, choose first move in remaining list



###############
## MAIN BODY ##
###############
def backgammon(score_to=1,whitestrat="GREEDY", weights1 = None, blackstrat="RANDOM", weights2 = None):
    """Play the backgammon game

    Args:
        score_to (int, optional): What score reached before terminate. Defaults to 1.
        writestrat (str, optional): Player 1 alg. Defaults to "GREEDY".
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
    w_score, b_score = 0,0
    p1vector = [0,0,0] 
    pminus1vector = [0,0,0] 
    
    game = 1
    #### MAIN LOOP ####
    while max([w_score, b_score]) < score_to:
        board = make_board()
        # board = [2]+[1]*13 + [-2]+ [-1]*13
        time_step = 1
        #### GAME LOOP ####
        while not game_over(board) and not is_error(board):
            if GUI_FLAG:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
            #### FIRST TURN ####            
            if time_step == 1 and game == 1:
                # Each player rolls a die to determine who moves first
                black_roll, white_roll = roll_dice()
                #### DISPLAY FIRST DICE ROLL FOR WHO GOES FIRST ####
                if GUI_FLAG:
                    # background = Background('Images/two_players_back.png')
                    # white_score = Shape('Images/White-score.png', SCREEN_WIDTH-36, SCREEN_HEIGHT//2 + 40)
                    # black_score = Shape('Images/Black-score.png', SCREEN_WIDTH-35, SCREEN_HEIGHT//2 - 40)
                    
                    pygame.display.update()
                    framesPerSec.tick(30)
                    update_screen(background, white_score, black_score, board, w_score, b_score, True)
                    pygame.display.update()
                    sleep(1)
                    for i in range(60):
                        black_roll, white_roll = roll_dice()
                        window.blit(black_dice[black_roll-1], (SCREEN_WIDTH//4-28, SCREEN_HEIGHT//2))
                        window.blit(white_dice[white_roll-1], (3*SCREEN_WIDTH//4+28, SCREEN_HEIGHT//2))
                        pygame.display.update()
                    
                while black_roll == white_roll:
                    black_roll, white_roll = roll_dice()
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
                    player2 = 1
                    player2strat = whitestrat
                    if GUI_FLAG:
                        background.render()
                        window.blit(black_dice[black_roll-1], (SCREEN_WIDTH//4-28, SCREEN_HEIGHT//2))
                        window.blit(black_dice[white_roll-1], (SCREEN_WIDTH//4+28, SCREEN_HEIGHT//2))
                else:
                    # White starts first
                    player1 = 1
                    player1strat = whitestrat
                    player2 = -1
                    player2strat = blackstrat
                    if GUI_FLAG:
                        background.render()
                        window.blit(white_dice[black_roll-1], (3*SCREEN_WIDTH//4-28, SCREEN_HEIGHT//2))
                        window.blit(white_dice[white_roll-1], (3*SCREEN_WIDTH//4+28, SCREEN_HEIGHT//2))
                
                if GUI_FLAG:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                    update_screen(background, white_score, black_score, board, w_score, b_score)
                    pygame.display.update()
                    sleep(1)
                # Initial roll made up of both starting dice
                roll = [black_roll, white_roll]
                moves1, boards1 = get_valid_moves(player1, board, roll)
                print_board(board)
                if commentary:
                    print(f"Player {player1} rolled {roll}")
            else:
                # All other rolls are generated on spot
                moves1, boards1, roll = start_turn(player1, board)
                
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
                        move, board = greedy_play(moves1, boards1, board, player1, roll)
                        move = move.pop()
                        
                    elif player1strat == "GENETIC":
                        pass
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
                                
                                # update_screen(background, white_score, black_score, board, w_score, b_score, True)
                                highlight_checker(end_checkers[i], end_point[i], "Images/white_pawn.png")
                                pygame.display.update()
                        update_screen(background, white_score, black_score, board, w_score, b_score, True)
                        pygame.display.update()
                        sleep(1)
                    #### END OF WHITE PLAYER 1'S TURN ####
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
                if USER_PLAY or GUI_FLAG:
                    sleep(1)
                
                #### BLACK PLAYER 2'S TURN ####
                
                moves2, boards2, roll = start_turn(player2, board)
                if len(moves2) > 0:
                    if player2strat == "USER":
                        move, board = human_play(moves2, boards2, board, roll, player2)
                    elif player2strat == "RANDOM":
                        board, move = randobot_play(roll, moves2, boards2)
                    elif player2strat == "GREEDY":
                        move, board = greedy_play(moves2, boards2, board, player2, roll)
                        move = move.pop()
                    elif player2strat == "GENETIC":
                        pass
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
                            
                            # update_screen(background, white_score, black_score, board, w_score, b_score, True)
                            highlight_checker(end_checkers[i], end_point[i], "Images/black_pawn.png")
                            pygame.display.update()
                    update_screen(background, white_score, black_score, board, w_score, b_score, True)
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
                        move, board = greedy_play(moves1, boards1, board, player1, roll)
                        move = move.pop()
                    elif player1strat == "GENETIC":
                        pass
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
                            
                            # update_screen(background, white_score, black_score, board, w_score, b_score, True)
                            highlight_checker(end_checkers[i], end_point[i], "Images/black_pawn.png")
                            pygame.display.update()
                            
                    update_screen(background, white_score, black_score, board, w_score, b_score, True)
                    pygame.display.update()
                    sleep(1)
                    
                #### END OF BLACK PLAYER 1'S TURN ####
                
                if commentary:
                    print_board(board)
                if USER_PLAY or GUI_FLAG:
                    sleep(1)
                if is_error(board):
                    sleep(10)
                    break
                if game_over(board):
                    break
                
                #### WHITE PLAYER 2'S TURN ####
                
                moves2, boards2, roll = start_turn(player2, board)
                if len(moves2) > 0:
                    if player2strat == "USER":
                        move, board = human_play(moves2, boards2, board, roll, player2)
                    elif player2strat == "RANDOM":
                        board, move = randobot_play(roll, moves2, boards2)
                    elif player2strat == "GREEDY":
                        move, board = greedy_play(moves2, boards2, board, player2, roll)
                        move = move.pop()
                    elif player2strat == "GENETIC":
                        pass
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
                        
                        # if len(set(start_checkers)) < len(start_checkers) or len(set(end_checkers)) < len(end_checkers):
                            # fix_same_checker(start_checkers, end_checkers)
                        for i in range(len(start_point)):
                            highlight_checker(start_checkers[i], start_point[i], "Images/white_highlight.png")
                            pygame.display.update()
                            sleep(1)
                            
                            highlight_checker(end_checkers[i], end_point[i], "Images/white_highlight.png")
                            pygame.display.update()
                            sleep(1)
                            
                            # update_screen(background, white_score, black_score, board, w_score, b_score, True)
                            highlight_checker(end_checkers[i], end_point[i], "Images/white_pawn.png")
                            pygame.display.update()
                    update_screen(background, white_score, black_score, board, w_score, b_score, True)
                    pygame.display.update()
                    sleep(1)
                
            #### END OF WHITE PLAYER 2'S TURN ####
            
            print_board(board)
            if is_error(board):
                sleep(10)
                break
            time_step +=1
            if USER_PLAY:
                sleep(1)
        #### CHECKS FOR GAME OVER AND WINNING POINTS ####
        if game_over(board):
            game += 1
            timestep(time_step)
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
            if board[26] == -15:
                player1 = -1
                player1strat = blackstrat
                player2 = 1
                player2strat = whitestrat
                b_score = pminus1vector[0] + 2*pminus1vector[1] + 3*pminus1vector[2]
            else:
                player1 = 1
                player1strat = whitestrat
                player2 = -1
                player2strat = blackstrat
                w_score = p1vector[0] + 2*p1vector[1] + 3*p1vector[2]
                
        #### CHECKS FOR GAME OVER AND WINNING POINTS ####
        
    return p1vector, w_score, pminus1vector, b_score


def collect_data(p1strat, pminus1strat, first_to):
    myFile = "./Data/greedydata.txt"
    
    for _ in range(100):
        dataFile = open(myFile, 'a')
        p1vector,w_score,pminus1vector,b_score= backgammon(5, "RANDOM",None, "RANDOM",None)
        dataFile.write(f"{w_score}, {b_score}\n")
        print(p1vector,w_score,pminus1vector,b_score)
        dataFile.close()
    

           
if __name__ == "__main__":
    if len(sys.argv[:]) > 1:
        if sys.argv[1] == "data":
            if len(sys.argv[:]) >= 3:
                collect_data(sys.argv[2], sys.argv[3], 5)
            else:
                collect_data("RANDOM",'RANDOM',5)
        # elif sys.argv[1] == 'time':
        #     collect_times("GREEDY", "GREEDY", 5)
        
    else:
        score_to = 5
        player1strat = "USER"
        playerminus1strat = "USER"
        weights1, weights2 = None, None
        if player1strat == "GENETIC":
            weights1 = genetic(50,100)
        if playerminus1strat == "GENETIC":
            weights2 = genetic(50,100)
        p1vector, w_score, pminus1vector, b_score = backgammon(score_to,player1strat,weights1,playerminus1strat,weights2)
        print(p1vector,pminus1vector)
            
            
                

            
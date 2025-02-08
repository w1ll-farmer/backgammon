from turn import *
from constants import *

def genetic_evaluate(board_before, board_after, player,
             weights):
    
    walled_off, walled_off_hit, borne_off_add, bear_off_points, hit_off_points, hit_off_mult, \
    exposed_hit, wall_blot_home_points, wall_mult, blot_mult, home_mult, blot_points, wall_points, \
    home_points, blot_diff_mult, wall_diff_mult, wall_maintain, blot_maintain = weights
    # print("walled off points",walled_off, walled_off_hit)
    """Gives a score to a move

    Args:
        move (_type_): _description_
        board_before (list(int)): The board at the start of the turn
        board_after (list(int)): The resulting board if the move is made
        player (int): Whether player is controlling white or black

    Returns:
        int: The score associated to the move
    """
    # Home information
    home_after = get_home_info(player, board_after)[1]
    home_before = get_home_info(player, board_before)[1]
    # Wall information
    walls_after = count_walls(board_after, player)
    walls_before = count_walls(board_before, player)
    wall_diff = walls_after - walls_before
    # Blot information
    player_blots_before = count_blots(board_before, player)
    player_blots_after = count_blots(board_after, player)
    blot_diff = player_blots_before - player_blots_after
    
    points = 0
    borne_off = 0
    
    # If home board is completely walled off
    if len([point for point in home_after if point*player > 1]) == 6 and not all_past(board_after):
        points += walled_off
        # Count how many opponent blots were hit in home
    num_home_opp_blots = len([i for i in home_before if player*i == -1])
    if num_home_opp_blots > 0:
        points += num_home_opp_blots + walled_off_hit
            
    # Check if piece(s) borne off
    if board_before[26] - board_after[26] > 0 or board_after[27] - board_before[27] > 0:
        borne_off = (board_before[26] - board_after[26]) + (board_after[27] - board_before[27])
    
    # Add how many opp pieces are on bar
    if player == 1 and board_after[24] < 0:
        borne_off += borne_off_add
    elif player == -1 and board_after[25] > 0:
        borne_off += borne_off_add
    points += borne_off
    
    # Bearing off pieces
    if board_after[26] < board_before[26] or board_after[27] > board_before[27]:
        
        points += bear_off_points
    if not all_past(board_after):
        points -= len([i for i in home_after if i == player])
        
        # Calc numbers of home hits
        num_home_opp_blots_before = count_blots(home_before, -player)
        num_home_opp_blots_after = count_blots(home_after, -player)
        points += (num_home_opp_blots_before - num_home_opp_blots_after)
        
        
    # If piece(s) hit off
    if board_after[24] < board_before[24] or board_after[25] > board_before[25]:
        if player_blots_before >= player_blots_after:
            points += hit_off_points + hit_off_mult*(player_blots_before - player_blots_after)
        else:
            # Left a piece exposed
            points += exposed_hit
    if wall_diff > 0:
        points += wall_mult*wall_diff_mult

    if not all_past(board_after):
        # Check if number of home walls have changed
        home_walls_after = count_walls(home_after, player)
        home_walls_before = count_walls(home_before, player)
        home_wall_diff = home_walls_after - home_walls_before
        # print(home_wall_diff)
        if wall_diff > 0 and blot_diff > 0 and home_wall_diff > 0:
            points += wall_blot_home_points + wall_mult*wall_diff + blot_mult*blot_diff + home_mult*home_wall_diff
        # Decrease in number of blots
        elif blot_diff > 0:
            points += blot_points + blot_mult*blot_diff
        # Increase in number of walls
        elif wall_diff > 0:
            points += wall_points + wall_mult*wall_diff
        # Increase in number of walls in home
        elif home_wall_diff > 0:
            points += home_points + home_mult*home_wall_diff
            
        # Made more blots
        if blot_diff < 0:
            points += blot_diff
        # Made fewer walls
        if wall_diff < 0:
            points += 0.8*wall_diff
    
    # Move all pieces past this turn  
    if not all_past(board_before) and all_past(board_after):
        points += 6
        
    if blot_diff < 0:
        points += blot_diff_mult*blot_diff
    if wall_diff < 0:
        points += wall_diff_mult*wall_diff
    
    if not all_past(board_after):
        # Maintain walls
        if wall_diff == 0:
            points += wall_maintain
        # No increase or decrease in blots
        if blot_diff == 0:
            points += blot_maintain
    return points



def tiebreak(boards, current_board, player):
    tiebreak_scores = [0]*len(boards)
    for board in range(len(boards)):
        # How many contiguous walls are present
        contiguous = 0
        max_contiguous = 0
        for point in boards[board]:
            if is_wall(point, player):
                contiguous +=1
            else:
                if contiguous > max_contiguous: max_contiguous = contiguous
                contiguous = 0
        
        if contiguous > max_contiguous: max_contiguous = contiguous
        tiebreak_scores[board] = max_contiguous
    boards_copy = [boards[board] for board in range(len(boards)) if tiebreak_scores[board] == max(tiebreak_scores)]
    if len(boards_copy) < len(boards) and test: print("Narrow 1")
    boards = boards_copy.copy()
    if test: print(tiebreak_scores)
    if len(boards) > 1:
        tiebreak_scores = [0]*len(boards)
        for board in range(len(boards)):
            # How many walls are in home
            tiebreak_scores[board] = len([point for point in get_home_info(player, boards[board])[1] if is_wall(point, player)])
        boards_copy = [boards[board] for board in range(len(boards)) if tiebreak_scores[board] == max(tiebreak_scores)]
        if len(boards_copy) < len(boards) and test: print("Narrow 2")
        boards = boards_copy.copy()
        if test: print(tiebreak_scores)
    if len(boards) > 1:
        tiebreak_scores = [0]*len(boards)
        for board in range(len(boards)):
            # Are there any blots?
            tiebreak_scores[board] = -len([point for point in boards[board] if point == player])
        boards_copy = [boards[board] for board in range(len(boards)) if tiebreak_scores[board] == max(tiebreak_scores)]
        if len(boards_copy) < len(boards) and test: print("Narrow 3")
        boards = boards_copy.copy()
        if test: print(tiebreak_scores)
    if len(boards) > 1:
        # Choose boards that have most advanced furthest-back piece
        tiebreak_scores = [0]*len(boards)
        for board in range(len(boards)):
            furthest_back = get_furthest_back(boards[board], player)
            if player == 1:
                furthest_back = 23-furthest_back
            tiebreak_scores[board] = furthest_back
        boards_copy = [boards[board] for board in range(len(boards)) if tiebreak_scores[board] == max(tiebreak_scores)]
        if len(boards_copy) < len(boards) and test: print("Narrow 4")
        boards = boards_copy.copy()
        if test: print(tiebreak_scores)
    if len(boards) > 1:
        # Choose boards that have minimum pips
        tiebreak_scores = [0]*len(boards)
        for board in range(len(boards)):
            tiebreak_scores[board] = -calc_pips(boards[board], player)
        boards_copy = [boards[board] for board in range(len(boards)) if tiebreak_scores[board] == max(tiebreak_scores)]
        if len(boards_copy) < len(boards) and test: print("Narrow 5")
        boards = boards_copy.copy()
        if test: print(tiebreak_scores)
    if len(boards) > 1:
        # Choose boards that maximum opponents pips
        tiebreak_scores = [0]*len(boards)
        for board in range(len(boards)):
            tiebreak_scores[board] = calc_pips(boards[board], -player)
        boards_copy = [boards[board] for board in range(len(boards)) if tiebreak_scores[board] == max(tiebreak_scores)]
        if len(boards_copy) < len(boards) and test: print("Narrow 6")
        boards = boards_copy.copy()
        if test: print(tiebreak_scores)
    if len(boards) > 1:
        tiebreak_scores = [0]*len(boards)
        for board in range(len(boards)):
            # Encourage moving far-back pieces to prevent blockages later on
            if player == 1:
                for i in range(24):
                    if did_move_piece(current_board[i], boards[board][i], player):
                        tiebreak_scores[board] += (current_board[i] - boards[board][i]) * i
                        
            else:
                for i in range(24):
                    if did_move_piece(current_board[i], boards[board][i], player):
                        j = 23- i
                        tiebreak_scores[board] += j * (boards[board][i] - current_board[i])
        if test: print(tiebreak_scores)            
                
        boards_copy = [boards[board] for board in range(len(boards)) if tiebreak_scores[board] == max(tiebreak_scores)]
        if len(boards_copy) < len(boards) and test:
            print("Narrow 7")
        boards = boards_copy.copy()
    boards = list(map(list, set(map(tuple, boards))))
    return boards
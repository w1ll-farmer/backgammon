from turn import *
from constants import *
from testfile import *
from genetic_agent import *
def easy_evaluate(move, board_before, board_after, player):
    score = 0
    for m in move:
        start, end = m
        if end == 26 or end == 27:
            # Prioritise bearing off
            score += 4
            
        elif board_before[end] == -player:
            # Next hitting off blots
            score += 3
            if abs(board_after[end]) >= 2:
                score += 0.25
            
        elif board_before[end] == player:
            if abs(board_before[start]) > 2:
                # Making 2 walls when were 1
                score += 2.5
            elif board_before[start] == player:
                # Removing blot
                score += 2
            else:
                # Pushing wall forward
                score += 1.5
                
        elif end in get_home_info(player, board_after)[0]:
            # Player moves into their home board
            score += 1
            
        if board_after[start] == player:
            # Player doesn't generate advantage and leaves a blot
            score -= 1
    return score

def evaluate(board_before, board_after, player,
             weights=None):
    if weights is None:
        walled_off=17
        walled_off_hit=5
        borne_off_add=1
        bear_off_points=13
        hit_off_points=11
        hit_off_mult=0.5
        exposed_hit=7
        wall_blot_home_points=9
        wall_mult=0.2
        blot_mult=0.3
        home_mult=0.1
        blot_points=9
        wall_points=8
        home_points=7
        blot_diff_mult=1
        wall_diff_mult=0.8
        wall_maintain=0.09
        blot_maintain=0.08
    else:
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
    
    else:
        # Bearing off pieces
        if board_after[26] < board_before[26] or board_after[27] > board_before[27]:
            
            points += bear_off_points
            if not all_past(board_after):
                points -= len([i for i in home_after if i == player])
                
                # Calc numbers of home hits
                num_home_opp_blots_before = count_blots(home_before, -player)
                num_home_opp_blots_after = count_blots(home_after, -player)
                points += (num_home_opp_blots_before - num_home_opp_blots_after)
        
        else:
            # If piece(s) hit off
            if board_after[24] < board_before[24] or board_after[25] > board_before[25]:
                if player_blots_before >= player_blots_after:
                    points += hit_off_points + hit_off_mult*(player_blots_before - player_blots_after)
                else:
                    # Left a piece exposed
                    points += exposed_hit
                if wall_diff > 0:
                    points += wall_mult*wall_diff_mult

            elif not all_past(board_after):
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
            elif not all_past(board_before) and all_past(board_after):
                points += 6
                
            else:
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
        max_contiguous = calc_prime(boards[board], player)
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

def invert_greedy(boards, current_board, player, weights, moves):
    inv_board, inv_board_afters, inv_player = check_inverted(current_board, boards, player)
    if weights is None:
        inv_scores = [evaluate(current_board, boards[i], player, weights) for i in range(len(moves))]
    else:
        inv_scores = [genetic_evaluate(current_board, boards[i], player, weights) for i in range(len(moves))]
    
    inv_board_afters = [invert_board(i) for i in inv_board_afters]
    inv_sorted_triplets = sorted(zip(inv_scores, inv_board_afters, moves), key=lambda x: x[0], reverse=True)

    inv_sorted_scores, inv_sorted_boards, inv_sorted_moves = zip(*inv_sorted_triplets)
    max_inv_score = [i for i in inv_sorted_scores if i==max(inv_scores)]
    if len(max_inv_score) > 1:
        if test:
            print("Equal inv boards")
            for i in range(len(max_inv_score)): print(inv_sorted_boards[i])
        inv_sorted_boards = [invert_board(i) for i in inv_sorted_boards]
        chosen_inv_boards = tiebreak(inv_sorted_boards[0:len(max_inv_score)], inv_board, inv_player)
        chosen_inv_boards = [invert_board(i) for i in chosen_inv_boards]
    return chosen_inv_boards
#### CHECK THIS ALL

if test:
    
    tiebreak([[0, 2, 0, 3, 2, 4, 0, 0, 0, 1, 0, -4, 3, 0, 0, 0, -2, 0, 0, -2, 0, -4, 0, -3, 0, 0, 0, 0], [0, 2, 1, 3, 2, 3, 0, 0, 0, 0, 0, -4, 4, 0, 0, 0, -2, 0, 0, -2, 0, -4, 0, -3, 0, 0, 0, 0]], [0, 1, 0, 3, 2, 5, 0, 0, 0, 0, 0, -4, 4, 0, 0, 0, -2, 0, 0, -2, 0, -4, 0, -3, 0, 0, 0, 0], 1)
    invert_greedy([[0, 2, 0, 3, 2, 4, 0, 0, 0, 1, 0, -4, 3, 0, 0, 0, -2, 0, 0, -2, 0, -4, 0, -3, 0, 0, 0, 0], [0, 2, 1, 3, 2, 3, 0, 0, 0, 0, 0, -4, 4, 0, 0, 0, -2, 0, 0, -2, 0, -4, 0, -3, 0, 0, 0, 0]], [0, 1, 0, 3, 2, 5, 0, 0, 0, 0, 0, -4, 4, 0, 0, 0, -2, 0, 0, -2, 0, -4, 0, -3, 0, 0, 0, 0], 1, None, [(1, 2), (2, 3)])
    print("Running black and white tests on evaluations")
    # Make complete home wall
    print(evaluate([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,-5,2,-2,0,-2,-2,-2,-2,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,-3,2,-2,-2,-2,-2,-2,-2,0,0,0,0], -1))
    
    print(evaluate([2,2,2,2,2,0,-2,5,-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [2,2,2,2,2,2,-2,3,-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 1))
    
    # Hit off piece and make complete home wall
    print(evaluate([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,-5,2,-2,1,-2,-2,-2,-2,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,-3,2,-2,-2,-2,-2,-2,-2,0,1,0,0], -1))
    
    print(evaluate([2,2,2,2,2,-1,-2,5,-12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [2,2,2,2,2,2,-2,3,-12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 1))
    
    # Bear off while maintain home wall
    print(evaluate([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,-2,2,-3,-2,-2,-2,-2,-2,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,-2,2,-2,-2,-2,-2,-2,-2,0,0,-1,0], -1))
    
    print(evaluate([2,2,2,2,2,3,-2,2,-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [2,2,2,2,2,2,-2,2,-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], 1))
    
    # Bear off without exposing
    print(evaluate([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,-2,-2,-3,2,-2,-2,-2,-2,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,-2,-2,-2,2,-2,-2,-2,-2,0,0,-1,0], -1))
    
    print(evaluate([4,2,2,-2,2,5,-2,0,-11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [3,2,2,-2,2,5,-2,0,-11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], 1))
    
    # Bear off and expose
    print(evaluate([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,-2,-2,-3,2,-2,-2,-2,-2,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,-2,-2,-3,2,-1,-2,-2,-2,0,0,-1,0], -1))
    
    print(evaluate([4,2,2,-2,2,5,-2,0,-11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [4,2,2,-2,1,5,-2,0,-11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], 1))
    
    # Hit without exposing
    print(evaluate([0,0,0,0,0,0,0,0,0,12,-2,-2,-3,1,-2,-2,-2,-2,2,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,12,-2,0,-3,-2,-2,-2,-2,-2,2,0,0,0,0,0,0,1,0,0], -1))
    
    print(evaluate([0,0,0,0,0,0,4,2,2,0,2,3,-3,-1,-11,2,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,4,2,2,0,2,3,-3,2,-11,0,0,0,0,0,0,0,0,0,-1,0,0,0], 1))
    
    # Increasing num walls
    print(evaluate([0,0,0,0,0,0,0,0,0,12,-2,0,-3,1,-4,-2,-2,-2,2,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,12,-2,0,-3,1,-2,-2,-2,-2,2,-2,0,0,0,0,0,0,0,0], -1))
    
    print(evaluate([0,0,0,0,0,0,4,2,2,0,2,3,-3,-1,-11,2,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,2,2,2,2,0,2,3,-3,-1,-11,2,0,0,0,0,0,0,0,0,0,0,0,0], 1))
    
    # Moving blot to wall
    print(evaluate([0,0,0,0,0,0,0,0,0,12,-2,-1,-2,1,-4,-2,-2,-2,2,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,12,-2, 0,-3,1,-4,-2,-2,-2,2,0,0,0,0,0,0,0,0,0], -1))
    
    print(evaluate([0,0,0,0,0,0,0,0,2,2,2,2,3,1,3,-3,-1,-11,2,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,2,2,2,2,4,0,3,-3,-1,-11,2,0,0,0,0,0,0,0,0,0], 1))
    
    # Pushing wall to home
    print(evaluate([0,0,0,0,0,0,0,0,0,12,-2,-1,-2,1,-4,-2,-2,-2,2,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,12,-2, 0,-3,1,-4,-2,-2,-2,2,0,0,0,0,0,0,0,0,0], -1))
    
    print(evaluate([0,0,0,0,0,0,0,0,2,2,2,2,3,1,3,-3,-1,-11,2,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,2,2,2,2,4,0,3,-3,-1,-11,2,0,0,0,0,0,0,0,0,0], 1))
    
    print("End of tests\n")
    
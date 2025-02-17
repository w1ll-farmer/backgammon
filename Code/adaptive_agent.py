from turn import *
from random import randint

def adaptive_midgame(moves, boards, player, player_score, opponent_score, cube_val, first_to, weights, roll):
    # Aims to play the move that boosts equity most
    best_equity = calc_advanced_equity(boards[0], player, player_score, opponent_score, cube_val, first_to)
    best_board = boards[0]
    best_move = moves[0]
    best_board_av_lookahead = None
    for board in boards:
        equity = calc_advanced_equity(board, player, player_score, opponent_score, cube_val, first_to, weights)
        if best_equity is None or best_equity < equity:
            best_equity = equity
            best_board = board
            best_move = moves[boards.index(best_board)]
        elif best_equity == equity:
            if best_board_av_lookahead is None:
                best_board_av_lookahead = get_lookahead(best_board, player, roll, 
                                                        player_score, opponent_score, cube_val,
                                                        first_to, weights)
            if best_board_av_lookahead != -999:   
                current_board_av_lookahead = get_lookahead(board, player, roll, player_score,
                                                        opponent_score, cube_val, first_to, weights)
                if current_board_av_lookahead < best_board_av_lookahead:
                    best_equity = equity
                    best_board = board
                    best_move = moves[boards.index(best_board)]
                    best_board_av_lookahead = current_board_av_lookahead
    return best_move, best_board

def calc_advanced_equity(board, player, player_score, opponent_score, cube_val, first_to, weights=None):
    
    if weights is None:
        pip_weight = 0.05  # Increase importance of race in later game
        blot_penalty = -0.2  # Harsher penalty for blots that can be hit
        prime_weight = 0.12  # Increased impact of prime structures
        home_board_weight = 0.08  # Stronger boards are more valuable
        opp_home_board_penalty = -0.12  # Opponent's home board makes hits worse
        gammon_weight = 0.25  # Increased weight for gammon potential
        bearing_off_weight = 0.18  # Make bearing off more valuable
        cube_volatility_weight = 0.15
        pip_diff_normaliser = 0.01
    else:
        pip_weight, blot_penalty, prime_weight, home_board_weight, opp_home_board_penalty, \
        gammon_weight, bearing_off_weight, cube_volatility_weight, pip_diff_normaliser = weights
    
    player_home = get_home_info(player, board)[1]
    opp_home = get_home_info(-player, board)[1]
    
    # PIP race adjusted for non-linear importance
    pip_adv = calc_pips(board, -player) - calc_pips(board, player)
    normalized_pip_adv = pip_adv *pip_diff_normaliser  # Normalize pip difference

    # Advanced blot assessment
    blot_equity = evaluate_blots(board, player)

    # Advanced prime assessment (longer primes count more)
    prime = calc_prime(board, player)
    strong_prime_bonus = 0.2 if prime >= 4 else 0  # Extra weight for 4+ primes

    # Improved home board strength evaluation
    home_strength = evaluate_home_board(board, player)
    opp_home_strength = evaluate_home_board(board, -player)

    # Gammon potential scaling with match play
    gammon_potential = calc_advanced_gammon_potential(board, player, player_home, opp_home, player_score, opponent_score)

    # Bearing off progress adjusted for checker distribution
    bear_off_progress = abs(board[int(26.5+(player/2))]) / 3

    # Cube leverage factor (reward positions where you have a strong double)
    cube_volatility = evaluate_cube_volatility(board, player, player_score, opponent_score, cube_val, first_to)

    # Compute final equity
    equity = (
        pip_weight * normalized_pip_adv +
        blot_equity * blot_penalty +
        prime_weight * prime + strong_prime_bonus +
        home_board_weight * home_strength +
        opp_home_board_penalty * opp_home_strength +
        gammon_weight * gammon_potential +
        bearing_off_weight * bear_off_progress +
        cube_volatility_weight * cube_volatility
    )

    
    return equity

def evaluate_cube_volatility(board, player, player_score, opponent_score, cube_val, first_to):
    base_volatility = calc_position_volatility(board, player)
    if player_score + cube_val >= first_to:
        return base_volatility * 1.5
    elif player_score < opponent_score -4:
        return base_volatility * 0.8
    else:
        return base_volatility
    
def calc_position_volatility(board, player):
    """
    Estimates the volatility of a position by considering:
      - The number of blots (higher volatility if many blots exist)
      - The presence of hit opportunities (both for and against the player)
      - The opponent’s home board strength (affecting re-entry difficulty)
      - The stage of the game (higher volatility in early/midgame)
    """
    num_blots = count_blots(board, player) + count_blots(board, -player)
    hit_chances = evaluate_blots(board, -player) - evaluate_blots(board, player)
    opp_home_strength = count_walls(get_home_info(-player, board)[1], -player)

    # Higher volatility if there are many blots, strong home boards, or high hitting chances
    volatility = (0.4 * num_blots) + (0.3 * abs(hit_chances)) + (0.3 * opp_home_strength)
    
    return min(1, volatility / 10)  # Normalize to [0,1] range

def calc_advanced_gammon_potential(board, player, player_home, opp_home, player_score, opponent_score):
    """
    Computes the potential for the player to win a gammon, incorporating match score considerations.

    Factors considered:
      - Opponent checkers in player's home board (trapped checkers)
      - Opponent checkers still in their home board (not yet moved out)
      - Player's progress toward bearing off
      - Match score (winning a gammon matters more in certain match situations)
    """
    # Count opponent checkers in different danger zones
    opp_pieces_home = count_walls(opp_home, -player) + count_blots(opp_home, -player)
    opp_pieces_player_home = count_walls(player_home, -player) + count_blots(player_home, -player)
    
    # Player's bearing-off progress (normalized)
    bearing_off_progress = abs(board[int(26.5 + (player / 2))]) / 3  # Normalized measure of bearing-off

    # Compute base gammon potential
    base_gammon_potential = (
        0.2 * opp_pieces_home + 
        0.3 * opp_pieces_player_home + 
        0.1 * bearing_off_progress
    )
    
    # **Match Score Factor:**
    # If the player benefits significantly from winning a gammon at this score, increase the weight.
    # If a gammon win is irrelevant (e.g., only 1 point needed to win the match), reduce weight.
    match_factor = compute_match_gammon_importance(player_score, opponent_score)
    
    # Adjust gammon potential based on match factor
    adjusted_gammon_potential = base_gammon_potential * match_factor
    
    # Ensure the result is between 0 and 1
    return min(1, adjusted_gammon_potential)

def compute_match_gammon_importance(player_score, opponent_score):
    """
    Computes how important a gammon win is based on match score.
    
    Key insights:
      - If the player is one point away from winning, a gammon is unnecessary (low importance).
      - If the opponent is close to winning the match, preventing a gammon loss is critical.
      - If the player is behind, winning a gammon can significantly improve chances.
      - If the player needs exactly 2 points to win, a gammon win secures the match (very high importance).
    """
    target_score = 25  # First-to-25 match

    # If winning a gammon wins the match, it's very important
    if player_score + 2 >= target_score:
        return 1.5  # Prioritize winning a gammon

    # If the player is far behind, gammon wins are highly valuable to catch up
    if player_score < opponent_score - 4:
        return 1.3

    # If the match is close, gammon importance is moderate
    if abs(player_score - opponent_score) <= 3:
        return 1.0

    # If a gammon win doesn't change the outcome much, reduce weight
    if player_score + 2 >= target_score:
        return 0.5  # Gammon doesn't matter if 1 point is enough

    return 1.0  # Default value for normal situations

def evaluate_home_board(board, player):
    """
    Evaluates the strength of the player's home board.
    Factors include: 
      - number of solid points or walls (anchors/prime segments)
      - number of blots in the home board (we want fewer)
      - difficulty for the opponent to reenter (if opponent checkers are on the bar)
    """
    player_home = get_home_info(player, board)[1]  # Assume returns a list/region of points in the home board
    
    # Count the number of 'walls' (consecutive occupied points) in the home board.
    wall_count = count_walls(player_home, player)
    # Count exposed blots in the home board.
    home_blots = count_blots(player_home, player)
    # Evaluate how hard it is for the opponent to reenter if they are on the bar.
    reentry_difficulty = evaluate_reentry_difficulty(player_home, player)
    
    # The home board score is higher with more walls, fewer blots, and greater reentry difficulty for the opponent.
    home_board_score = (0.10 * wall_count) - (0.08 * home_blots) + (0.12 * reentry_difficulty)
    return home_board_score

def evaluate_reentry_difficulty(home_points, player):
    # Returns a score indicating how difficult it is for the opponent to reenter.
    # For example, count how many points are controlled (occupied by 2 or more checkers)
    difficulty = 0
    for point in home_points:
        if (player == 1 and point >= 2) or (player == -1 and point <= -2):
            difficulty += 1
    return difficulty / 6

def evaluate_blots(board, player):
    """
    Returns a blot score that penalizes exposed checkers and rewards checkers that are protected.
    Also, additional penalty is applied if a blot is within the opponent's hitting range.
    """
    exposed = 0
    vulnerability_penalty = 0
    safe = 0
    for point in range(24):
        if board[point] == player:
            # Check if the blot is in danger: in range of an opponent's anchor.
            p_hit = prob_opponent_can_hit(player, board,board[point])
            vulnerability_penalty += p_hit
            if not adjacent_friend(board, point, player) and p_hit > 0:
                exposed +=1
            else:
                safe += 1

    # Combine measures into a blot equity score.
    # (Weights are tunable; here we penalize exposed blots heavily,
    # reward safe positions slightly, and add extra penalty for vulnerable blots.)
    blot_score = (-0.25 * exposed) + (0.10 * safe) - (0.20 * vulnerability_penalty)
    return blot_score

def adjacent_friend(board, point, player):
    if player == 1 and board[point-1] >= player or board[point+1] >= player:
        return True
    elif player == -1 and board[point-1] <= player or board[point+1] <= player:
        return True
    return False

def prob_opponent_can_hit(player, board, point):
    start_points = [i for i in range(len(board)) if (player == 1 and board[i] < 0) or (player == -1 and board[i] > 0)]
    can_hit = 0
    for roll1 in range(1,7):
        for roll2 in range(1, 7):
            for s in start_points:
                if s + (roll1+roll2)*(1+roll1==roll2) <= point:
                    if s + roll1 == point or s + roll2 == point:
                        can_hit +=1
                    elif roll1 == roll2:
                        if s + (roll1*2) == point or s+(roll2)*2 == point or s + (roll1*2)+roll2 == point or s + (roll2*2)+roll1 == point:
                            can_hit +=1
    return can_hit/36

def get_lookahead(start_board, player, roll, player_score, opponent_score, cube_val, first_to, weights):
    equities = []
    for roll1 in range(1, 7):
        for roll2 in range(1, 7):
            moves, boards = get_valid_moves(player, start_board, roll)
            for board in boards:
                equities.append(calc_advanced_equity(board, player, player_score, opponent_score, cube_val, first_to, weights))
    return sum(equities)/len(equities) if len(equities) > 0 else -999
            
def race_gwc(board, player):
    # Source: https://bkgm.com/articles/Matussek/BearoffGWC.pdf?utm_source=chatgpt.com
    player_home = get_home_info(player, board)[1]
    opp_home = get_home_info(-player, board)[1]
    epc1 = calc_epc(board, player)
    epc2 = calc_epc(board, -player)
    coefficient = get_coefficient(epc1) + classify_bearoff_position(player_home, opp_home)
    gwc = 0.5 + coefficient*(epc2 - epc1 + 4.1)
    return gwc

def adaptive_race(moves, boards, player):
    best_move = None
    best_gwc = None
    best_board = None
    for board in boards:
        gwc = race_gwc(board, player)
        if best_gwc is None or gwc > best_gwc:
            best_gwc = gwc
            best_board = board
            best_move = moves[boards.index(best_board)]
    return best_move, best_board
    

def calc_epc(board, player):
    home = get_home_info(player, board)[1]
    wastage = 4.56 + 1.95*abs(home[0]) + 1.31*abs(home[1]) + 0.84*abs(home[2]) + 0.4*abs(home[3]) + 0.23*abs(home[4]) + 0.11*abs(home[5])
    epc = calc_pips(board, player) + wastage
    if abs(home[0]) > 0 and count_walls(home, player) + count_blots(home, player) >= 4:
        epc -=1
    if any([i >= 4 for i in home]):
        epc +=1
    if any([i >=6 for i in home]):
        epc +=1
    return epc

def get_coefficient(epc1):
    if epc1 >= 104.7: return 2
    elif epc1 >= 74.5: return 2.5
    elif epc1 >= 56.4: return 3
    elif epc1 >= 44.6: return 3.5
    elif epc1 >= 36.3: return 4
    elif epc1 >= 30.4: return 4.5
    elif epc1 >= 25.9: return 5
    elif epc1 >= 22.4: return 5.5
    else: return 19.4

def is_roll_position(distribution):
    low_points_total = distribution[0] + distribution[1] + distribution[2]
    if low_points_total < 5:
        return True
    else:
        return False

def classify_bearoff_position(dist_on_roll, dist_off_roll):
    """
    Interpretation:
      - "pip-pip": Neither player's ability to bear off depends much on the next roll.
      - "roll-pip": The player on roll is in a roll-dependent position but the opponent is not.
      - "pip-roll": The opponent is roll-dependent while the player on roll is not.
      - "roll-roll": Both players are in positions that depend strongly on the next roll.
    """
    # Use the heuristic function to determine roll dependence.
    on_roll_is_roll = is_roll_position(dist_on_roll)
    off_roll_is_roll = is_roll_position(dist_off_roll)
    
    if not on_roll_is_roll and not off_roll_is_roll:
        return 0
    elif on_roll_is_roll and not off_roll_is_roll:
        return 0.5
    elif not on_roll_is_roll and off_roll_is_roll:
        return 0.5
    else:
        return 1

def move_furthest_back(player, current_board, moves, boards):
    scores = [0]*len(boards)
    for board in range(len(boards)):
            # Encourage moving far-back pieces to prevent blockages later on
            if player == 1:
                for i in range(24):
                    if did_move_piece(current_board[i], boards[board][i], player):
                        scores[board] += (current_board[i] - boards[board][i]) * i
                        
            else:
                for i in range(24):
                    if did_move_piece(current_board[i], boards[board][i], player):
                        j = 23- i
                        scores[board] += j * (boards[board][i] - current_board[i])
    boards_copy = [boards[board] for board in range(len(boards)) if scores[board] == max(scores)]
    if len(boards_copy) == 1:
        chosen_board = boards_copy[0]
    else:
        chosen_board = boards_copy[randint(0, len(boards_copy)-1)]
    chosen_move = moves[boards.index(chosen_board)]
    return chosen_move, chosen_board
        
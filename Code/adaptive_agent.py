from turn import *
from random import randint


def adaptive_midgame(moves, boards, player, player_score, opponent_score, cube_val, first_to, weights, roll):
    # Aims to play the move that boosts equity most
    
    equities = [calc_advanced_equity(board, player, player_score, opponent_score, cube_val, first_to, weights) for board in boards]
    best_boards = [boards[i] for i in range(len(boards)) if equities[i] == max(equities)]
    if len(best_boards) > 1 and not is_double(roll):
        lookahead_equities = [get_lookahead(board, player, roll, player_score, opponent_score, cube_val, first_to, weights) for board in best_boards]
        best_boards = [best_boards[i] for i in range(len(best_boards)) if lookahead_equities[i] == min(lookahead_equities)]
    
    if len(best_boards) > 1:
        chosen_board = best_boards[randint(0, len(best_boards))-1]
        chosen_move = moves[boards.index(chosen_board)]
    else:
        chosen_board = best_boards[0]
        chosen_move = moves[boards.index(chosen_board)]
    
    return chosen_move, chosen_board

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
        prime_bonus = 0.2
        bear_off_normaliser = 0.33
        win_volatility = 1.5
        catch_up_volatility=0.8
        blot_volatility=0.4
        hit_volatility=0.3
        opp_home_volatility=0.3
        opp_piece_home_potential=0.2
        opp_piece_player_potential=0.3
        bear_off_potential=0.1
        win_bonus=1.5
        catch_up_bonus=1.3
        close_bonus=1.0
        no_chance_bonus=0.5
        default=1.0
        wall_mult=0.1
        home_mult=-0.08
        reentry_mult=0.12
        exposed_mult=0.25
        safe_mult=0.1
        vulnerable_mult=0.2
        volatility_normaliser =0.1
    else:
        pip_weight, blot_penalty, prime_weight, home_board_weight, opp_home_board_penalty, \
        gammon_weight, bearing_off_weight, cube_volatility_weight, pip_diff_normaliser, \
        prime_bonus, bear_off_normaliser,  win_volatility, catch_up_volatility, \
        blot_volatility, hit_volatility, opp_home_volatility, opp_piece_home_potential, \
        opp_piece_player_potential, bear_off_potential, win_bonus, catch_up_bonus, \
        close_bonus, no_chance_bonus, default, wall_mult, home_mult, reentry_mult, \
        exposed_mult, safe_mult, vulnerable_mult, volatility_normaliser = weights
    
    player_home = get_home_info(player, board)[1]
    opp_home = get_home_info(-player, board)[1]
    
    # PIP race adjusted for non-linear importance
    pip_adv = calc_pips(board, -player) - calc_pips(board, player)
    normalized_pip_adv = pip_adv *pip_diff_normaliser  # Normalize pip difference

    # Advanced blot assessment
    blot_equity = evaluate_blots(board, player, exposed_mult, safe_mult, vulnerable_mult)

    # Advanced prime assessment (longer primes count more)
    prime = calc_prime(board, player)
    strong_prime_bonus = prime_bonus if prime >= 4 else 0  # Extra weight for 4+ primes

    # Improved home board strength evaluation
    home_strength = evaluate_home_board(board, player, wall_mult, home_mult, reentry_mult)
    opp_home_strength = evaluate_home_board(board, -player, wall_mult, home_mult, reentry_mult)

    # Gammon potential scaling with match play
    gammon_potential = calc_advanced_gammon_potential(board, player, player_home, opp_home,
                                                      player_score, opponent_score, first_to,
                                                      bear_off_normaliser, opp_piece_home_potential,
                                                      opp_piece_player_potential, bear_off_potential, 
                                                      win_bonus, catch_up_bonus, close_bonus, no_chance_bonus, default)

    # Bearing off progress adjusted for checker distribution
    bear_off_progress = abs(board[int(26.5+(player/2))]) * bear_off_normaliser

    # Cube leverage factor (reward positions where you have a strong double)
    cube_volatility = evaluate_cube_volatility(board, player, player_score, opponent_score,
                                               cube_val, first_to, win_volatility,
                                               catch_up_volatility, blot_volatility, hit_volatility,
                                               opp_home_volatility, volatility_normaliser,
                                               exposed_mult, safe_mult, vulnerable_mult)

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

def evaluate_cube_volatility(board, player, player_score, opponent_score, cube_val, first_to,
                             win_volatility, catch_up_volatility, blot_volatility,
                             hit_volatility, opp_home_volatility, volatility_normaliser,
                             exposed_mult, safe_mult, vulnerable_mult):
    
    base_volatility = calc_position_volatility(board, player, blot_volatility,
                                               hit_volatility, opp_home_volatility, volatility_normaliser,
                                               exposed_mult, safe_mult, vulnerable_mult)
    if player_score + cube_val >= first_to:
        return base_volatility * win_volatility
    elif player_score < opponent_score -4:
        return base_volatility * catch_up_volatility
    else:
        return base_volatility
    
def calc_position_volatility(board, player, blot_volatility, hit_volatility,
                             opp_home_volatility, volatility_normaliser, exposed_mult,
                             safe_mult, vulnerable_mult):
    """
    Estimates the volatility of a position by considering:
      - The number of blots (higher volatility if many blots exist)
      - The presence of hit opportunities (both for and against the player)
      - The opponent’s home board strength (affecting re-entry difficulty)
      - The stage of the game (higher volatility in early/midgame)
    """
    num_blots = count_blots(board, player) + count_blots(board, -player)
    hit_chances = evaluate_blots(board, -player, exposed_mult, safe_mult, vulnerable_mult) \
        - evaluate_blots(board, player, exposed_mult, safe_mult, vulnerable_mult)
    opp_home_strength = count_walls(get_home_info(-player, board)[1], -player)
    # Higher volatility if there are many blots, strong home boards, or high hitting chances
    volatility = (blot_volatility * num_blots) + (hit_volatility * abs(hit_chances)) + (opp_home_volatility * opp_home_strength)
    
    return min(1, volatility * volatility_normaliser)  # Normalize to [0,1] range

def calc_advanced_gammon_potential(board, player, player_home, opp_home, player_score, opponent_score, first_to,
                                   bear_off_normaliser, opp_piece_home_potential,
                                   opp_piece_player_potential, bear_off_potential,
                                   win_bonus, catch_up_bonus, close_bonus,
                                   no_chance_bonus, default):
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
    bearing_off_progress = abs(board[int(26.5 + (player / 2))]) * bear_off_normaliser  # Normalized measure of bearing-off
    # Compute base gammon potential
    base_gammon_potential = (
        opp_piece_home_potential * opp_pieces_home + 
        opp_piece_player_potential * opp_pieces_player_home + 
        bear_off_potential * bearing_off_progress
    )
    
    # **Match Score Factor:**
    # If the player benefits significantly from winning a gammon at this score, increase the weight.
    # If a gammon win is irrelevant (e.g., only 1 point needed to win the match), reduce weight.
    match_factor = compute_match_gammon_importance(player_score, opponent_score, first_to, win_bonus,
                                                   catch_up_bonus, close_bonus, no_chance_bonus, default)
    
    # Adjust gammon potential based on match factor
    
    adjusted_gammon_potential = base_gammon_potential * match_factor
    
    # Ensure the result is between 0 and 1
    return min(1, adjusted_gammon_potential)

def compute_match_gammon_importance(player_score, opponent_score, first_to,
                                    win_bonus, catch_up_bonus, close_bonus,
                                    no_chance_bonus, default):
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
        return win_bonus  # Prioritize winning a gammon

    # If the player is far behind, gammon wins are highly valuable to catch up
    if player_score < opponent_score - 4:
        return catch_up_bonus

    # If the match is close, gammon importance is moderate
    if abs(player_score - opponent_score) <= 3:
        return close_bonus

    # If a gammon win doesn't change the outcome much, reduce weight
    if player_score + 2 >= target_score:
        return no_chance_bonus  # Gammon doesn't matter if 1 point is enough

    return default  # Default value for normal situations

def evaluate_home_board(board, player, wall_mult, home_mult, reentry_mult):
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
    home_board_score = (wall_mult * wall_count) + (home_mult * home_blots) + (reentry_mult * reentry_difficulty)
    return home_board_score

def evaluate_reentry_difficulty(home_points, player):
    # Returns a score indicating how difficult it is for the opponent to reenter.
    # For example, count how many points are controlled (occupied by 2 or more checkers)
    difficulty = 0
    for point in home_points:
        if (player == 1 and point >= 2) or (player == -1 and point <= -2):
            difficulty += 1
    return difficulty / 6

def evaluate_blots(board, player, exposed_mult, safe_mult, vulnerable_mult):
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
    blot_score = (-exposed_mult * exposed) + (safe_mult * safe) - (vulnerable_mult * vulnerability_penalty)
    return blot_score

def adjacent_friend(board, point, player):
    if player == 1 and board[point-1] >= player or board[point+1] >= player:
        return True
    elif player == -1 and board[point-1] <= player or board[point+1] <= player:
        return True
    return False

def get_lookahead(start_board, player, roll, player_score, opponent_score, cube_val, first_to, weights):
    equities = []
    for roll1 in range(1, 7):
        for roll2 in range(1, 7):
            moves, boards = get_valid_moves(player, start_board, [roll1, roll2])
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


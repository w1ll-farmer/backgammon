from turn import *
def adaptive_play(moves, boards, player, roll, board):
    # Aims to play the move that boosts equity most
    pass

def calc_equity(board, player):
    # Assign heuristic weights (tuned via testing)
    pip_weight = 0.02
    blot_penalty = -0.1
    prime_weight = 0.08
    home_board_weight = 0.05
    opponent_home_board_penalty = -0.1
    gammon_weight = 0.2
    bearing_off_weight = 0.15

    player_home = get_home_info(player, board)[1]
    opp_home = get_home_info(-player, board)[1]
    pip_adv = calc_pips(board, -player) - calc_pips(board, player)
    blots = count_blots(board, player) + count_blots(board, -player)
    prime = calc_prime(board, player)
    home_strength = count_walls(player_home) - 0.5 * count_blots(player_home)
    opp_home_strength = count_walls(opp_home) - 0.5 * count_blots(opp_home)
    gammon_potential = calc_gammon_potential(board, player, player_home, opp_home)
    bear_off_progress = abs(board[int(26.5+(player/2))]) / 15
    
    equity = (
        pip_weight * pip_adv +
        blot_penalty * blots +
        prime_weight * prime +
        home_board_weight * home_strength +
        opponent_home_board_penalty * opp_home_strength +
        gammon_weight * gammon_potential +
        bearing_off_weight * bear_off_progress
    )
    # Clamp equity to [-1, 1] range
    return max(-1, min(1, equity))

def calc_gammon_potential(board, player, player_home, opp_home):
    opp_pieces_home = count_walls(opp_home, -player) + count_blots(opp_home, -player)
    opp_pieces_player_home = count_walls(player_home, -player) + count_blots(player_home, -player)
    bearing_off_progress = abs(board[int(26.5+(player/2))]) / 15
    gammon_potential = (
        0.2 * opp_pieces_home +
        0.3 * opp_pieces_player_home +
        0.1 * bearing_off_progress
    )
    return  min(1, gammon_potential)

"""
function calculate_equity(board_state):

    # Step 5: Calculate gammon risk for the opponent (based on back checkers)
    gammon_potential = calculate_gammon_potential(board_state)

    # Step 6: Calculate how many checkers the player has borne off
    bearing_off_progress = calculate_bearing_off_progress(board_state)

    # Step 7: Compute equity using weighted values for the factors
    equity = (
        pip_weight * pip_advantage + 
        blot_penalty * blots + 
        prime_weight * prime_strength + 
        home_board_weight * home_board_strength + 
        opponent_home_board_penalty * opponent_home_board_strength + 
        gammon_weight * gammon_potential + 
        bearing_off_weight * bearing_off_progress
    )

    # Step 8: Limit equity to be within the range [-1, 1] (ensuring valid output)
    equity = max(-1, min(1, equity))

    return equity


# Supporting Functions:

# Function to calculate gammon potential (number of opponent's far-back checkers)
function calculate_gammon_potential(board_state):
    return count_opponent_checkers_far_from_home(board_state)

# Function to calculate bearing off progress
function calculate_bearing_off_progress(board_state):
    return count_checkers_bearing_off(board_state, player)


# Constants (weights for each factor can be tuned based on testing)
pip_weight = 0.02
blot_penalty = -0.1
prime_weight = 0.08
home_board_weight = 0.05
opponent_home_board_penalty = -0.1
gammon_weight = 0.2
bearing_off_weight = 0.15

"""
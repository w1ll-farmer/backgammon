from turn import *
import struct
def adaptive_midgame(moves, boards, player, turn):
    # Aims to play the move that boosts equity most
    
    best_equity = None
    best_board = None
    best_move = None
    for board in boards:
        equity = calc_equity(board, player)
        if best_equity is None or best_equity < equity:
            best_equity = equity
            best_board = board
            best_move = moves[boards.index(best_board)]

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
    home_strength = count_walls(player_home, player) - 0.5 * count_blots(player_home, player)
    opp_home_strength = count_walls(opp_home, player) - 0.5 * count_blots(opp_home, player)
    gammon_potential = calc_gammon_potential(board, player, player_home, opp_home)
    bear_off_progress = abs(board[int(26.5+(player/2))]) / 3
    
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
    bearing_off_progress = abs(board[int(26.5+(player/2))])
    gammon_potential = (
        0.2 * opp_pieces_home +
        0.3 * opp_pieces_player_home +
        0.1 * bearing_off_progress
    )
    return  min(1, gammon_potential)

def race(board, player):
    player_home = get_home_info(player, board)[1]
    opp_home = get_home_info(-player, board)[1]
    epc1 = calc_epc(board, player)
    epc2 = calc_epc(board, -player)
    coefficient = get_coefficient(epc1) + classify_bearoff_position(player_home, opp_home)
    gwc = 0.5 + coefficient*(epc2 - epc1 + 4.1)
    return gwc
    
    

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


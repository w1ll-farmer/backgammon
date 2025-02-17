from constants import *
from random import randint
from turn import *
from adaptive_agent import calc_advanced_equity
crawford_game = False

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

def calc_equity(board, player):
    # Assign heuristic weights (tuned via testing)
    pip_weight = 0.02
    blot_penalty = -0.15
    prime_weight = 0.08
    home_board_weight = 0.05
    opponent_home_board_penalty = -0.1
    gammon_weight = 0.2
    bearing_off_weight = 0.15

    player_home = get_home_info(player, board)[1]
    opp_home = get_home_info(-player, board)[1]
    pip_adv = calc_pips(board, -player) - calc_pips(board, player)
    blots = count_blots(board, player) - 0.5* count_blots(board, -player)
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

def can_double(double_player, current_player):
    if crawford_game: return False
    elif double_player == 0: return True
    elif double_player == current_player: return True
    else: return False
    
def double(cube_val, player):
    return cube_val*2, -player

def randobot_accept_double():
    return bool(randint(0,1))

def basic_should_double(equity):
    return True if equity > 0.5 else False

def basic_accept_double(equity):
    return True if equity > -0.5 else False

def user_accept_double(player, cube_val, double_player):
    user_accept = input("Opponent offer x2. y/n").lower()
    if user_accept == 'y':
        cube_val, double_player = double(cube_val, player)
        has_double_rejected = False
    else:
        has_double_rejected = True
    return cube_val, double_player, has_double_rejected


def is_crawford_game(w_score, b_score, score_to, prev_score):
    if prev_score[0] == score_to - 1 or prev_score[1] == score_to - 1:
        return False
    elif w_score == score_to - 1 or b_score == score_to - 1:
        return True
    else:
        return False
    
def get_double_rejected_board(player):
    return [int(0.5-(player/2)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,int(-0.5-(player/2)),0,0,int(-14.5+(player/2)),int(14.5+(player/2))]

def double_process(playerstrat, player, board, oppstrat, cube_val, double_player, player_score, opponent_score, first_to):
    has_double_rejected = False
    if player in strategies:
        if basic_should_double(calc_equity(board, player)):
            # Player doubles opponenent
            if oppstrat in strategies: 
                if oppstrat == "ADAPTIVE":
                    if basic_accept_double(calc_advanced_equity(board, player, player_score, opponent_score, cube_val, first_to)):
                        cube_val *= 2
                        double_player = -player
                elif basic_accept_double(calc_equity(board, -player)):
                    # Opponent accepts double
                    cube_val *= 2
                    double_player = -player
                else:
                    has_double_rejected = True
            elif oppstrat == "USER":
                if not GUI_FLAG:
                    cube_val, double_player, has_double_rejected = user_accept_double(-player, cube_val, double_player)
                else:
                    print("Feature is work in progress")
    else:
        if playerstrat == "USER":
            if not GUI_FLAG:
                print("Would you like to double your opponent?")
                user_double = input("y/n ").lower()
                if user_double == "y":
                    if oppstrat == "USER":
                        cube_val, double_player, has_double_rejected = user_accept_double(-player, cube_val, double_player)
                    elif oppstrat in strategies:
                        if oppstrat == "ADAPTIVE":
                            if basic_accept_double(calc_advanced_equity(board, player, player_score, opponent_score, cube_val, first_to)):
                                cube_val *= 2
                                double_player = -player
                        elif basic_accept_double(calc_equity(board, -player)):
                            # Opponent accepts double
                            cube_val *= 2
                            double_player = -player
                        else:
                            has_double_rejected = True
                    elif oppstrat == "RANDOM":
                        if randobot_accept_double():
                            cube_val *= 2
                            double_player = -player
                        else:
                            has_double_rejected = True
            else:
                print("Feature not yet implemented")
    return cube_val, double_player, has_double_rejected
# Before dice roll, player has option to double
# If player choooses to double, opponent may reject or accept
# If reject, player wins point
# If accept, game is now twice as valuable. Opponent is only one who can double
# End of game, cube is reset
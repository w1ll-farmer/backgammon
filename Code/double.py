crawford_game = False
def can_double(double_player, current_player):
    if double_player == 0: return True
    elif double_player == current_player: return True
    else: return False
    
def double(cube_val):
    return cube_val*2

def basic_should_double(equity):
    return True if equity > 0.5 else False

def basic_accept_double(equity):
    return True if equity > -0.5 else False

def is_crawford_game(p1score, p2score, score_to):
    if score_to -1 == p1score and score_to-1 ==p2score:
        return False
    elif score_to-1 == p1score: 
        return True
    elif score_to-1 == p2score:
        return
    else:
        return False
from constants import *
from random import randint
from turn import *
from adaptive_agent import calc_advanced_equity, race_gwc
from gui import *
from testfile import invert_board

import torch
import torch.nn as nn
import torch.nn.functional as F 
if GUI_FLAG:
    import pygame
    from pygame.locals import *
    
class AcceptNet(nn.Module):
    def __init__(self):
        super(AcceptNet, self).__init__()
        self.fc1 = nn.Linear(289, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class OfferNet(nn.Module):
    def __init__(self):
        super(OfferNet, self).__init__()
        self.fc1 = nn.Linear(289, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

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

def can_double(double_player, current_player, w_score, b_score, score_to, prev_score):
    # return False
    if is_crawford_game(w_score, b_score, score_to, prev_score):
        return False
    elif double_player == 0:
        return True
    elif double_player == current_player:
        return True
    else:
        return False
    
def double(cube_val, player):
    return cube_val*2, -player

def randobot_accept_double():
    return bool(randint(0,1))

def basic_should_double(equity):
    return True if equity > 0.8144 else False

def basic_accept_double(equity):
    return True if equity > -0.2362 else False

def advanced_should_double(equity, doubling_point = 1.4325859937671366): 
    if doubling_point is None: doubling_point = 2.8334
    return True if equity > doubling_point else False

def advanced_accept_double(equity, doubling_point = -1.8523842372779313):
    if doubling_point is None: doubling_point = -0.3126
    return True if equity > doubling_point else False

def deep_accept_double(board, player):
    if player == -1:
        board = invert_board(board)
    input_vector = torch.tensor(convert_board(board), dtype=torch.float32).unsqueeze(0)
    model = AcceptNet()
    model.load_state_dict(torch.load(os.path.join("Code","cube_accept_model.pth")))
    model.eval()
    with torch.no_grad():
        decision = model(input_vector).item()  # Get the single output
    return (decision > 0.5)

def deep_offer_double(board, player):
    if player == -1:
        board = invert_board(board)
    input_vector = torch.tensor(convert_board(board), dtype=torch.float32).unsqueeze(0)
    model = OfferNet()
    model.load_state_dict(torch.load(os.path.join("Code","cube_offer_model.pth")))
    model.eval()
    with torch.no_grad():
        decision = model(input_vector).item()  # Get the single output
    return (decision > 0.5)

def user_accept_double(player, cube_val, double_player):
    if not GUI_FLAG:
        user_accept = input("Opponent offer x2. y/n").lower()
        if user_accept == 'y':
            cube_val, double_player = double(cube_val, player)
            has_double_rejected = False
        else:
            has_double_rejected = True
    else:
        double_accept_window = Shape(white_dice_paths[-1], SCREEN_WIDTH//2, SCREEN_HEIGHT//2, 240,160)
        double_accept_window.draw(window)
        double_accept_window.addText(window, "Opponent offers x2", black)
        cross = Shape(cross_path, SCREEN_WIDTH//2 - 60, SCREEN_HEIGHT//2 + 40, 48, 48)
        tick = Shape(tick_path, SCREEN_WIDTH//2 + 60, SCREEN_HEIGHT//2 + 40, 48, 48)
        cross.draw(window)
        tick.draw(window)
        pygame.display.update()
        clicked = False
        while not clicked:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    
                    click = pygame.mouse.get_pos()
                    if tick.rect.collidepoint(click):
                        has_double_rejected = False
                        cube_val, double_player = double(cube_val, player)
                        
                        clicked = True
                    if cross.rect.collidepoint(click):
                        has_double_rejected = True
                        clicked = True
                        
               
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

def accept_process(board, player, player_score, oppstrat, opponent_score, first_to, cube_val, double_player):
    has_double_rejected = False
    if oppstrat in strategies:
        if oppstrat == "ADAPTIVE":
            if advanced_accept_double(calc_advanced_equity(board, player, player_score, opponent_score, cube_val, first_to)):
                cube_val *= 2
                double_player = -player
            else:
                has_double_rejected = True
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
    elif oppstrat == "DEEP":
        if deep_accept_double(board, -player):
            cube_val *= 2
            double_player = -player
        else:
            has_double_rejected = True
    elif oppstrat == "USER":
        cube_val, double_player, has_double_rejected = user_accept_double(player, cube_val, double_player)
    return cube_val, double_player, has_double_rejected
                            
def double_process(playerstrat, player, board, oppstrat, cube_val, double_player, player_score, opponent_score, first_to, double_point=None, double_drop=None):
    has_double_rejected = False
    double_offered = False
    gwc = -1
    if playerstrat in strategies:
        if playerstrat == "ADAPTIVE":
            if all_checkers_home(player, board) and all_past(board) and abs(board[int(26.5+0.5*player)]) >= 7:
                gwc = race_gwc(board, player)
            equity = calc_advanced_equity(board, player, player_score, opponent_score, cube_val, first_to, [0.9966066885314592, -0.9916984096898946, 0.3106830724424913, 0.529168163359478, -0.4710732676896102, 0.5969523488654117, 0.36822981983332415, 0.38958074063216697, 0.02676397245530815, 0.08588282381449319, 0.06094873757931751, 1.1095422351658368, 0.47764793610307643, 0.040753486445243126, 0.5495226441839489, 0.8875009606764003, 0.9333344067224983, 0.1340269726805713, 0.1978868967026618, 1.2096547126804458, 2.379707426788366, 0.6465298771549699, 0.509196585225148, 0.261875669397977, 0.36883752029556166, -0.481342015629518, 0.7098436807557322, 1.0250219115287624, 0.5739284594183071, 0.1796876959733017, 0.2679991261065485])
        else:
            equity = calc_equity(board, player)
        if basic_should_double(equity) and playerstrat != "ADAPTIVE" or \
            advanced_should_double(equity, double_point) and playerstrat == "ADAPTIVE" and gwc < 0 or \
                playerstrat == "ADAPTIVE" and gwc > 0.8:
            # Player doubles opponenent
            cube_val, double_player, has_double_rejected = accept_process(
                board, player, player_score, oppstrat, opponent_score, first_to, cube_val, double_player
            )
    else:
        if playerstrat == "USER":
            if not GUI_FLAG:
                print("Would you like to double your opponent?")
                user_double = input("y/n ").lower()
                if user_double == "y":
                    cube_val, double_player, has_double_rejected = accept_process(
                        board, player, player_score, oppstrat, opponent_score, first_to, cube_val, double_player
                        )
            else:
                cube = display_double_cube(player, cube_val)
                pygame.display.update()
                clicked = False
                while clicked == False:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            click = pygame.mouse.get_pos()
                            if cube.rect.collidepoint(click):
                                double_offered = True
                            clicked = True
                if double_offered:
                    cube_val, double_player, has_double_rejected = accept_process(
                        board, player, player_score, oppstrat, opponent_score, first_to, cube_val, double_player
                    )
                    
                            
        elif playerstrat == "DEEP":
            if deep_offer_double(board, player):
                cube_val, double_player, has_double_rejected = accept_process(
                    board, player, player_score, oppstrat, opponent_score, first_to, cube_val, double_player
                )
                
                        
    return cube_val, double_player, has_double_rejected

"""Best Doubling points from 4x25 testing:
(77, ' 1.4114902594531387', ' 0.9329566864476466') -> 200-130 = 70, 467-285=182 (15-5)

(62, ' 1.7000678176853496', ' -0.08512374869267081') -> 241-172 = 69, 496-274=222 (18-2)
(57, ' 1.4325859937671366', ' -1.8523842372779313') -> 234-150 = 84, 507-264 =243 (19-1)

1.7000678176853496, -0.08512374869267081 -> 1209-817 = 392 (42-8)
1.4325859937671366, -1.8523842372779313 -> 1210-765 = 445 (40-10)
"""
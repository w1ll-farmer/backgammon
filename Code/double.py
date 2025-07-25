from constants import *
from random import randint
from turn import *
from adaptive_agent import calc_advanced_equity, race_gwc
from gui import *
from testfile import invert_board
from reinforce_agent import ReinforceNet
from reinforce_play import encode_state
import torch
import torch.nn as nn
import torch.nn.functional as F 
from gnubg_interact import gnubg_accept_double, gnubg_offer_double
if GUI_FLAG:
    import pygame
    from pygame.locals import *
    
class AcceptNet(nn.Module):
    def __init__(self):
        super(AcceptNet, self).__init__()
        self.fc1 = nn.Linear(291, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class OfferNet(nn.Module):
    def __init__(self):
        super(OfferNet, self).__init__()
        self.fc1 = nn.Linear(291, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
class RaceAcceptNet(nn.Module):
    def __init__(self):
        super(RaceAcceptNet, self).__init__()
        self.fc1 = nn.Linear(265, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class RaceOfferNet(nn.Module):
    def __init__(self):
        super(RaceOfferNet, self).__init__()
        self.fc1 = nn.Linear(265, 12)
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

def can_double(double_player, current_player, w_score, b_score, score_to, prev_score, cube_on=True):
    if (w_score == 24 and current_player == -1) or (b_score == 24 and current_player == 1):
        return False # If it rejects double it loses match.
    if not cube_on:
        return False
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
    return True if equity > 0.9828 else False

def basic_accept_double(equity):
    return True if equity > -0.35 else False

def advanced_should_double(equity, doubling_point = 3.4233): 
    doubling_point = 3.8498771759475505
    # doubling_point = 3.4233
    return True if equity > doubling_point else False

def advanced_accept_double(equity, doubling_point=2.4609298125822554):
    doubling_point = 2.4609298125822554
    # if doubling_point is None: doubling_point = -0.3126
    # doubling_point = -0.6252
    return True if equity > doubling_point else False

def deep_accept_double(board, player, race):
    if player == -1:
        board = invert_board(board)
    encoded_board = convert_board(board, race=race, cube=True)
    input_vector = torch.tensor(encoded_board, dtype=torch.float32).unsqueeze(0)
    if race:
        model = RaceAcceptNet()
        model.load_state_dict(torch.load(os.path.join("Code","race_cube_accept_model.pth")))
    else:
        model = AcceptNet()
        model.load_state_dict(torch.load(os.path.join("Code","cube_accept_model.pth")))
    model.eval()
    with torch.no_grad():
        decision = model(input_vector).item()  # Get the single output
        # print(decision)
    return (decision > 0.5)
# deep_accept_double([0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-2,-3,-1,-1,0,0,-7,13], 1, True)
def deep_offer_double(board, player, race):
    if player == -1:
        board = invert_board(board)
    encoded_board = convert_board(board, race=race, cube=True)
    input_vector = torch.tensor(encoded_board, dtype=torch.float32).unsqueeze(0)
    if race:
        model = RaceOfferNet()
        model.load_state_dict(torch.load(os.path.join("Code","race_cube_offer_model.pth")))
    else:
        model = OfferNet()
        model.load_state_dict(torch.load(os.path.join("Code","cube_offer_model.pth")))
    model.eval()
    with torch.no_grad():
        decision = model(input_vector).item()  # Get the single output
        # print(decision)
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

def reinforce_accept_double(board, player, ep, double_point = 0.2):
    """Uses RL model to make accept decision

    Args:
        board (list(int)): Raw board
        player (int): The player being doubled
        ep (str): The episode being used as agent
        double_point (float, optional): Folding point. Defaults to 0.8.

    Returns:
        bool: 1 for accept 0 for reject
    """
    if player == -1:
        board = invert_board(board)
    model = ReinforceNet()
    if ep[0] == "O":
        model.load_state_dict(torch.load(os.path.join("Code","RL","One",f"reinforcement_{ep[3:]}.pth"))['model_state_dict'])
    else:
        model.load_state_dict(torch.load(os.path.join("Code","RL",f"reinforcement_{ep}.pth"))['model_state_dict'])
    # encoded_boards = [encode_state(board) for board in inverted_boards]
    board_tensor = torch.FloatTensor(np.array([encode_state(board, 1)]))
    with torch.no_grad():
        outcome_probs = model(board_tensor)  # Shape: [num_boards, 6]
        # expected_vals = model.expected_value(outcome_probs)
    # print(outcome_probs[0])
    win, loss, gwin, gloss, bgwin, bgloss = outcome_probs[0]
    win = win.item()
    gwin = gwin.item()
    bgwin = bgwin.item()
    loss = loss.item()
    # PHASE 1
    # if win > double_point:
    #     return True
    # else:
    #     return False
    # PHASE 2
    if win - loss > model.expected_value(outcome_probs) or sum([win,gwin,bgwin]) > 0.2: # cubeless > cubeful?:
        return False
    else:
        return True
    
def reinforce_should_double(board, player, ep):
    if player == -1:
        board = invert_board(board)
    model = ReinforceNet()
    if ep[0] == "O":
        model.load_state_dict(torch.load(os.path.join("Code","RL","One",f"reinforcement_{ep[3:]}.pth"))['model_state_dict'])
    else:
        model.load_state_dict(torch.load(os.path.join("Code","RL",f"reinforcement_{ep}.pth"))['model_state_dict'])
    # encoded_boards = [encode_state(board) for board in inverted_boards]
    board_tensor = torch.FloatTensor(np.array([encode_state(board, 1)]))
    with torch.no_grad():
        outcome_probs = model(board_tensor)  # Shape: [num_boards, 6]
        expected_val = model.expected_value(outcome_probs)
    win, loss, gwin, _, bgwin, _ = outcome_probs[0]
    win = win.item()
    gwin = gwin.item()
    bgwin = bgwin.item()
    loss = loss.item()
    # Phase 1
    if sum([win, gwin, bgwin]) > 0.7:
        if expected_val > 2: # position too good to double
            return False
        else:
            return True
    else:
        return False
    # # PHASE 2
    # if win - loss < expected_val:# and expected_val < 2:
    #     return True
    # else:
    #     return False


def is_crawford_game(w_score, b_score, score_to, prev_score):
    if prev_score[0] == score_to - 1 or prev_score[1] == score_to - 1:
        return False
    elif w_score == score_to - 1 or b_score == score_to - 1:
        return True
    else:
        return False
    
def get_double_rejected_board(player):
    return [int(0.5-(player/2)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,int(-0.5-(player/2)),0,0,int(-14.5+(player/2)),int(14.5+(player/2))]

def accept_process(board, player, player_score, oppstrat, opponent_score, first_to, cube_val, double_player, ep="self_170000"):
    if player_score + cube_val > first_to:
        return cube_val*2, -player, False
    has_double_rejected = False
    if oppstrat in strategies:
        if oppstrat == "ADAPTIVE":
            # Does this need to be changed to opponent instead of player?
            if advanced_accept_double(calc_advanced_equity(board, -player, opponent_score, player_score, cube_val, first_to)):
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
            
    elif oppstrat == "DEEP" or oppstrat == "REINFORCEMENT":
        race = all_past(board) or board[26] < 0 or board[27] > 0
        if deep_accept_double(board, -player, race=race):
            cube_val *= 2
            double_player = -player
        else:
            has_double_rejected = True
            
    elif oppstrat == "USER":
        cube_val, double_player, has_double_rejected = user_accept_double(player, cube_val, double_player)
    
    elif oppstrat == "GNUBG":
        if gnubg_accept_double(board, cube_val, player): # Don't negate player
            cube_val *=2
            double_player = -player
        else:
            has_double_rejected = True
    else:
        print("Unidentified strategy")
    return cube_val, double_player, has_double_rejected
                            
def double_process(playerstrat, player, board, oppstrat, cube_val, double_player, player_score, opponent_score, first_to, double_point=None, double_drop=None, ep="self_170000"):
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
                    
                            
        elif oppstrat == "DEEP" or oppstrat == "REINFORCEMENT":
            race = all_past(board) or board[26] < 0 or board[27] > 0
            if deep_offer_double(board, player, race=race):
                cube_val, double_player, has_double_rejected = accept_process(
                    board, player, player_score, oppstrat, opponent_score, first_to, cube_val, double_player
                )
        
        elif playerstrat == "REINFORCEMENT":
            if reinforce_should_double(board, player, ep):
                cube_val, double_player, has_double_rejected = accept_process(
                    board, player, player_score, oppstrat, opponent_score, first_to, cube_val, double_player
                )      
        
        elif playerstrat == "GNUBG":
            if gnubg_offer_double(board, cube_val, player):
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
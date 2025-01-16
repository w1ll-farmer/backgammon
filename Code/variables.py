from constants import *
if GUI_FLAG:
    background = Background('Images/two_players_back.png')
    white_score = Shape('Images/White-score.png', SCREEN_WIDTH-36, SCREEN_HEIGHT//2 + 40)
    black_score = Shape('Images/Black-score.png', SCREEN_WIDTH-35, SCREEN_HEIGHT//2 - 40)
    w_score, b_score = 0,0
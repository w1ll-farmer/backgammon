import os
SCREEN_WIDTH, SCREEN_HEIGHT = 900, 790
commentary = False
USER_PLAY = False
GUI_FLAG = False
black = [0,0,0]
white = [255,255,255]
FPS = 30
train = False
if USER_PLAY == True: commentary = True
if GUI_FLAG == True: commentary = False

strategies = ["GREEDY","GENETIC","EXPECTIMAX","ADAPTIVE"]

test=False
# GRAPHICS INITIALISATION
if GUI_FLAG:
    import pygame
    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    framesPerSec = pygame.time.Clock()
    pygame.display.set_caption("Backgammon")
    window.fill(black)
    
    cross_path = os.path.join("Images","cross.png")
    tick_path = os.path.join("Images","tick.png")
    
    white_dice_paths = [
        os.path.join("Images","player_dice1.png"),
        os.path.join("Images","player_dice2.png"),
        os.path.join("Images","player_dice3.png"),
        os.path.join("Images","player_dice4.png"),
        os.path.join("Images","player_dice5.png"),
        os.path.join("Images","player_dice6.png"),
        os.path.join("Images","white_doubling_cube.png"),
        os.path.join("Images","blank_white_dice.png")
    ]
    black_dice_paths = [
        os.path.join("Images","adversary_dice1.png"),
        os.path.join("Images","adversary_dice2.png"),
        os.path.join("Images","adversary_dice3.png"),
        os.path.join("Images","adversary_dice4.png"),
        os.path.join("Images","adversary_dice5.png"),
        os.path.join("Images","adversary_dice6.png"),
        os.path.join("Images","black_doubling_cube.png"),
        os.path.join("Images","blank_black_dice.png")
    ]
    white_dice, black_dice = [], []
    for i in range(8):
        white_dice.append(pygame.image.load(white_dice_paths[i]))
        black_dice.append(pygame.image.load(black_dice_paths[i]))
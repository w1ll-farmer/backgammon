import pygame
SCREEN_HEIGHT, SCREEN_WIDTH = 800, 800
commentary = False
USER_PLAY = False
RANDOM_PLAY = False
GREEDY_PLAY = True
GUI_FLAG = False
black = [0,0,0]
white = [255,255,255]
FPS = 30

if USER_PLAY == True: commentary = True


# GRAPHICS INITIALISATION
if GUI_FLAG:
    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    framesPerSec = pygame.time.Clock()
    
    pygame.display.set_caption("Backgammon")
    window.fill(black)
import pygame
from constants import *
from turn import roll_dice, update_board
from time import sleep
class Background: #creates a background
    def __init__(self,backgroundImage):
        # Sets background to passed in image
        self.backgroundImage = pygame.image.load(backgroundImage)
        self.backgroundImage = pygame.transform.scale(self.backgroundImage, (SCREEN_WIDTH, SCREEN_HEIGHT))
        # Creates a rectangle around it  for co-ordinates
        self.backgroundRect = self.backgroundImage.get_rect()
        
        # Sets X co-ordinates
        self.backgroundX1 = 0 
        # Sets Y co-ordinates
        self.backgroundY1 = (SCREEN_HEIGHT-self.backgroundRect.height)//4 
        
    def render(self): 
        # Renders in the background
        window.blit(self.backgroundImage, (self.backgroundX1, self.backgroundY1))
        
    def update(self,backgroundImage): 
        # Updates the background image
        self.backgroundImage = pygame.image.load(backgroundImage)
        self.backgroundRect = self.backgroundImage.get_rect()

class Shape: 
    # An image or shape that can have text added to it
    def __init__(self,image, x, y, width=60, height=48):
        self.image = pygame.image.load(image)
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(center = (x, y))
        self.font = pygame.font.SysFont('Calibri', 20)
        
    def move(self, x, y): 
        # Adjust coordinates of shape
        self.rect=self.image.get_rect(center=(x,y))
        
    def draw(self,window): 
        # Draws Shape onto screen
        window.blit(self.image,self.rect)
        
    def addText(self, window, text, colour):
        # Adds text to centre of object
        text_surface = self.font.render(text, True, colour)
        text_rect = text_surface.get_rect()
        # Center the text inside the Shape's rect
        text_rect.center = self.rect.center
        window.blit(text_surface, text_rect)
        

def get_top_row_checker_pos(point, checker):
    # Returns checker's coordinates for display
    offset_x = 88
    offset_y = 51
    if point > 5:
        offset_x += 56
    return (offset_x + (point*56), offset_y+(checker*56))

def get_bottom_row_checker_pos(point, checker):
    # Returns checker's coordinates for display
    offset_x = SCREEN_WIDTH - 141
    offset_y = 93
    if point > 5:
        offset_x -= 56
    return (offset_x - (point*56), SCREEN_HEIGHT-(offset_y+(checker*56)))

def get_white_home_pos(checker):
    # Returns white borne-off checker's coordinates for display
    return (839, 743 - (checker * 12))  

def get_black_home_pos(checker):
    # Returns black borne-off checker's coordinates for display
    return (839, 43 + (checker * 12))

def display_board(board):
    for point in range(0,12):
        for checker in range(abs(board[point])):
            if board[point] > 0:
                window.blit(pygame.image.load("Images/white_pawn.png"), get_bottom_row_checker_pos(point, checker))
            elif board[point] < 0:
                window.blit(pygame.image.load("Images/black_pawn.png"),get_bottom_row_checker_pos(point, checker))
    
    for point in range(12, 24):        
        for checker in range(abs(board[point])):
            if board[point] > 0:
                window.blit(pygame.image.load("Images/white_pawn.png"), get_top_row_checker_pos(point-12, checker))
            elif board[point] < 0:
                window.blit(pygame.image.load("Images/black_pawn.png"),get_top_row_checker_pos(point-12, checker))
    
    if board[24] < 0:
        black_bar_checker = Shape("Images/black_pawn.png", SCREEN_WIDTH//2 + 3,SCREEN_HEIGHT//2 - 40, 56, 56)
        black_bar_checker.draw(window)
        if board[24] < -1:
            black_bar_checker.addText(window, f"{abs(board[24])}", white)
    
    if board[25] > 0:
        white_bar_checker = Shape("Images/white_pawn.png", SCREEN_WIDTH//2 + 3,SCREEN_HEIGHT//2 + 40, 56, 56)
        white_bar_checker.draw(window)
        if board[25] > 1:
            white_bar_checker.addText(window, f"{board[25]}", black)
    
    if board[26] < 0:
        for checker in range(-board[26]):
            window.blit(pygame.image.load("Images/black_pawn_outside.png"), get_black_home_pos(checker))
    
    if board[27] > 0:
        for checker in range(board[27]):
            window.blit(pygame.image.load("Images/white_pawn_outside.png"), get_white_home_pos(checker))
    
    pygame.display.update()
    
    
def display_dice_roll(colour):
    # Displays animation for rolling dice
    if colour == 1:
        mult = 3
        dice_list = white_dice
    else:
        mult = 1
        dice_list = black_dice
    for i in range(60):
        die1, die2 = roll_dice()
        window.blit(dice_list[die1-1], (mult*SCREEN_WIDTH//4-28, SCREEN_HEIGHT//2))
        window.blit(dice_list[die2-1], (mult*SCREEN_WIDTH//4+28, SCREEN_HEIGHT//2))
        pygame.display.update()
    sleep(1)
    return die1, die2


def fix_same_checker(start_points, start_checkers, end_points, end_checkers):
    
    new_start, new_end = start_checkers, end_checkers
    for i in range(len(start_points)):
        # dupes = sum([j for j in range(i+1, len(start_points)) if start_points[i] == start_points[j] and start_checkers[i] == start_checkers[j]])
        for j in range(i+1, len(start_points)):
            if start_points[i] == start_points[j]:
                if start_checkers[i] == start_checkers[j]:
                    new_start[i] +=1
            if end_points[i] == end_points[j]:
                if end_checkers[i] == end_checkers[j]:
                    new_end[i] -= 2
    
    return new_start, new_end


def parse_move(board, move):
    # Extracts which checker is being moved and to where for animation 
    start_point, end_point = [], []
    print(move)
    for m in move:
        start_point.append(m[0])
        end_point.append(m[1])
    start_checkers = []
    end_checkers = []
    for point in range(len(start_point)):
        print("start point",start_point)
        start_checkers.append(abs(board[start_point[point]]))
        end_checkers.append(abs(board[end_point[point]])-1)
    start_checkers, end_checkers = fix_same_checker(start_point, start_checkers, end_point, end_checkers)
    return start_point, start_checkers, end_point, end_checkers


# print(fix_same_checker([23, 21, 23, 23], [3, 3, 3, 3], [19, 20, 21, 18], [3,3,3,3])) 


        
def highlight_checker(checker, point, img_path):
    if checker < 0: checker = 0
    # Highlights checker to show move 
    if point < 12:
        window.blit(pygame.image.load(img_path), get_bottom_row_checker_pos(point, checker))
        
    elif point < 24:
        window.blit(pygame.image.load(img_path), get_top_row_checker_pos(point-12, checker))
        
    elif point < 26:
        if "white" in img_path:
            white_bar_checker_highlight = Shape(img_path, SCREEN_WIDTH//2 + 3,SCREEN_HEIGHT//2 + 40, 56, 56)
            white_bar_checker_highlight.draw(window)
        else:
            black_bar_checker_highlight = Shape(img_path, SCREEN_WIDTH//2 + 3,SCREEN_HEIGHT//2 - 40, 56, 56)
            black_bar_checker_highlight.draw(window)
        
def update_screen(background, white_score, black_score, board, w_score, b_score, include_bground=False):
    if include_bground:
        background.render()
    white_score.draw(window)
    white_score.addText(window, f'{w_score}/5',black)
    black_score.draw(window)
    black_score.addText(window, f'{b_score}/5',white)
    display_board(board)
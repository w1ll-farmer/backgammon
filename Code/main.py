import pygame
from pygame.locals import *
from random_agent import *
from turn import *
import numpy as np
from time import sleep 
from greedy_agent import *
global GUI_FLAG
global USER_PLAY
global SCREEN_HEIGHT
global SCREEN_WIDTH
global FPS
global window
global framesPerSec
global black
global white 

GUI_FLAG = False
USER_PLAY = False
if GUI_FLAG:
    SCREEN_HEIGHT, SCREEN_WIDTH = 800, 800
    black = [0,0,0]
    white = [255,255,255]
    pygame.init()
    FPS = 30
    framesPerSec = pygame.time.Clock()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Backgammon")
    window.fill(black)

class Background: #creates a background
    def __init__(self,backgroundImage):
        # super().__init__()
        # Sets background to passed in image
        self.backgroundImage = pygame.image.load(backgroundImage)
        self.backgroundImage = pygame.transform.scale(self.backgroundImage, (800, 700))
        # Creates a rectangle around it  for co-ordinates
        self.backgroundRect = self.backgroundImage.get_rect()
        
        # Sets X co-ordinates
        self.backgroundX1 = 0 
        # Sets Y co-ordinates
        self.backgroundY1 = (SCREEN_HEIGHT-self.backgroundRect.height)//4 
        # self.backgroundY1 = 0
        
    def render(self): #Renders in the background
        window.blit(self.backgroundImage, (self.backgroundX1, self.backgroundY1))
        
    def update(self,backgroundImage): #Updates the background image
        self.backgroundImage = pygame.image.load(backgroundImage)
        self.backgroundRect = self.backgroundImage.get_rect()

class Shape: #Same as box but takes on an image instead of a colour
    def __init__(self,image,x,y, width=60, height=48):
        self.image = pygame.image.load(image)
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(center = (x,y))
        self.font = pygame.font.SysFont('Calibri',20)
    def move(self,window,x,y): #moves Shape
        self.rect=self.image.get_rect(center=(x,y))
    def draw(self,window): #displays Shape
        window.blit(self.image,self.rect)
    def addText(self,window,text,colour):
        text_surface = self.font.render(text, True, colour)
        
        # Get the text's rect
        text_rect = text_surface.get_rect()

        # Center the text inside the Shape's rect
        text_rect.center = self.rect.center
        
        # Blit the text onto the window
        window.blit(text_surface, text_rect)
        
        
def start_turn(player, board):
    """Rolls the dice and finds all possible moves.

    Args:
        player (int): The encoding of the player, -1 or 1
        board ([int]): The representation of the board

    Returns:
        ([(int, int)], [int], [int]): Possible moves, associated boards, diceroll
    """
    roll = roll_dice()
    print(f"Player {player} rolled {roll}")
    moves, boards = get_valid_moves(player, board, roll)
    return moves, boards, roll

def human_play(moves, boards):
    """Lets the human player make a move

    Args:
        moves ([(int, int)]): Possible start-end pairs.
        boards ([[int]]): All boards associated with each move

    Returns:
        [int]: The resulting board after making the move
    """
    if len(moves) > 0:
        moves_stringified = [str(move1) for move1 in moves]
        move = input("Enter move.")
        # Loop until valid move is selected
        while move not in moves_stringified:
            print("Enter a valid move")
            move = input("")
        move_index = moves_stringified.index(move)
        board = boards[move_index]
    else:
        print("No valid moves available")
    return board

def randobot_play(roll, moves, boards):
    """Random agent makes a move

    Args:
        roll ([int]): The dice roll.
        moves ([[(int, int)]]): List of all possible start-end pairs
        boards ([[int]]): The boards associated to each move

    Returns:
        [int], [(int, int)]: The resulting board and move chosen
    """
    move = []
    attempts = 0
    # Repeats until 200,000 random moves have been chosen
    # Or until a valid move has been selected
    while move not in moves and attempts < 200000:
        move = []
        for _ in range(1+is_double(roll)):
            move.append(generate_random_move())
            move.append(generate_random_move())
        attempts += 1
    # In case no random move was valid
    if attempts == 200000:
        print('no found moves')
    if move not in moves:
        if len(moves) > 1:
            move = moves[randint(0, len(moves)-1)]
        elif len(moves) == 1:
            move = moves[0]
    
        
    board = boards[moves.index(move)]
    return board, move

def greedy_play(moves, boards, current_board, player):
    scores = [evaluate(moves[i], current_board, boards[i], player) for i in range(len(moves))]
    sorted_pairs = sorted(zip(scores, boards), key=lambda x: x[0], reverse=True)
    sorted_scores, sorted_boards = zip(*sorted_pairs)
    return list(sorted_boards)[0]
    

###############
## MAIN BODY ##
###############
def backgammon():
    if GUI_FLAG == True:
        background_board = Background('Images/board_unaltered.png')
        white_score = Shape('Images/White-score.png', 38, SCREEN_HEIGHT-150)
        black_score = Shape('Images/Black-score.png', 40, 150)
        white_checker = Shape('Images/black_pawn.png', 200, 200, 42, 42)
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
            pygame.display.update()
            framesPerSec.tick(30)
            background_board.render()
            
            white_score.draw(window)
            white_score.addText(window, '0/5',black)
            
            black_score.draw(window)
            black_score.addText(window, '0/5',white)
            white_checker.draw(window)
    # Each player rolls a die to determine who moves first
    black_roll, white_roll = roll_dice()
    # Loops in scenario rolls are equal
    while black_roll == white_roll:
        black_roll, white_roll = roll_dice()
        
    print(f"Black rolled {black_roll}")
    print(f"White rolled {white_roll}")
    # Black starts first
    if black_roll > white_roll:
        print("Computer starts")
        player1 = -1
        player2 = 1
    else:
        # White starts first
        print("User Starts")
        player1 = 1
        player2 = -1
        
    # running = True
    time_step = 1
    board = make_board()
    while not game_over(board):
        print(time_step)
        if time_step == 1:
            # Initial roll made up of both starting dice
            roll = [black_roll, white_roll]
            moves1, boards1 = get_valid_moves(player1, board, roll)
            print_board(board)
            print(f"Player {player1} rolled {roll}")
        else:
            # All other rolls are generated on spot
            moves1, boards1, roll = start_turn(player1, board)
        if USER_PLAY:
            sleep(0.5)
        if player1 == 1:
            if USER_PLAY:
                board = human_play(moves1, boards1)
            else:
                board, move = randobot_play(roll, moves1, boards1)
                print(f"Move Taken: {move}")
            print_board(board)
            
            # Game ends?
            if is_error(board):
                sleep(2)
                break
            if game_over(board):
                break
            if USER_PLAY:
                sleep(1)
            
            # Player 2's turn
            
            moves2, boards2, roll = start_turn(player2, board)
            board, move = randobot_play(roll, moves2, boards2)
            print(f"Move Taken: {move}")
        else:
            board, move = randobot_play(roll, moves1, boards1)
            print(f"Move Taken: {move}")
            print_board(board)
            if USER_PLAY:
                sleep(1)
            if is_error(board):
                sleep(2)
                break
            if game_over(board):
                break
            # Player 2 turn
            moves2, boards2, roll = start_turn(player2, board)
            if USER_PLAY:
                board = human_play(moves2, boards2)
            else:
                board, move = randobot_play(roll, moves2, boards2)
                print(f"Move Taken: {move}")
        print_board(board)
        if is_error(board):
            sleep(2)
            break
        time_step +=1
        if USER_PLAY:
            sleep(1)
    if game_over(board):
        print("GAME OVER")
        if board[26] == -15:
            print("Player -1 win")
        else:
            print("Player 1 win")
        if is_backgammon:
            print("By backgammon")
        elif is_gammon:
            print("By gammon")
            
if __name__ == "__main__":         
    backgammon()
            
            
                

            
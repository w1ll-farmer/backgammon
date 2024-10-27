import ai
import turn
import win
import numpy as np
from random import randint

# class Player:
#     def __init__(self, colour):
#         """Constructor for Player class. Sets default attribute values

#         Args:
#             colour (str): The colour of the player's pieces
#         """
#         self.colour = colour
#         self.unplaced = 0
#         self.placed = 15
#         self.beared_off = 0
    
#     def enter(self):
#         """Lets the player enter a checker back onto the board
#         """
#         self.placed += 1
#         self.unplaced -= 1
        
#     def hit(self):
#         """Sends checker to the bar
#         """
#         self.placed -= 1
#         self.unplaced += 1

#     def bear_off(self):
#         """Lets a player send a checker home
#         """
#         self.placed -= 1
#         self.beared_off += 1
        
# class Checker(Player):
#     def __init__(self, colour, board):
#         self.colour = colour
#         self.hit = False
#         self.beared_off = False
#         if self.colour == 'W':
#             self.home_cords = [1,6]
#             self.opp_home_cords = [0,6]
#         else:
#             self.home_cords = [0,6]
#             self.opp_home_cords[1,6]
#         self.home = board[self.home_cords[0]][self.home_cords[1]:]
#         self.opp_home = board[self.opp_home_cords[0]][self.opp_home_cords[1]:]
#         # print(self.home, self.opp_home)
#     def can_enter(self, roll):
#         valid = []
#         for die in roll:
#             if len(self.opp_home[-die][0]) < 2:
#                 print('less than 2',self.opp_home[-die][0])
#                 valid.append(die)
#             elif self.opp_home[-die][0][0] == self.colour:
#                 print('occupied by same',self.opp_home[-die][0][0])
#                 valid.append(die)
#         return valid

  
def print_board(board):
    for i in range(0,24,12):
        print(board[i:i+12])
        
#White home board -> black home board
# 24 is black bar, 25 is white bar
# 26 is black 'beard off' and 27 is white 'beard off'
board = [-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0,0,0,0]


inp = input("Black, White, or random? b/w/default=random").upper()

if inp == 'B' or inp == 'W': colour = inp
else: colour = ['B','W'][randint(0,1)]

# print(len(['BBB'].pop())) 
# Initialise player objects and board object
# player = Player(colour)
# cpu = Player([c for c in ['B','W'] if c != colour])
# # board.show()
# c = Checker('W',board.board)
# print(c.can_enter([1,6]))


import ai
import turn
import win
from random import randint

class Player:
    def __init__(self, colour):
        """Constructor for Player class. Sets default attribute values

        Args:
            colour (str): The colour of the player's pieces
        """
        self.colour = colour
        self.unplaced = 0
        self.placed = 15
        self.beared_off = 0
    
    def enter(self):
        """Lets the player enter a checker back onto the board
        """
        self.placed += 1
        self.unplaced -= 1
        
    def hit(self):
        """Sends checker to the bar
        """
        self.placed -= 1
        self.unplaced += 1

    def bear_off(self):
        """Lets a player send a checker home
        """
        self.placed -= 1
        self.beared_off += 1
        
class Board:
    def __init__(self):
        self.board = [
                    [[['WWWWW'],[],[],[],['BBB'],[]],[['BBBBB'],[],[],[],[],['WW']]],
                    [[['BBBBB'],[],[],[],['WWW'],[]],[['WWWWW'],[],[],[],[],['BB']]]
                    ]
    
    def show(self):
        for i in range(0, len(self.board)):
            print(self.board[i])
    
    


colour_list = ['Black', 'White']
colourDict = {'b':"Black", 'w':'White'}
inp = input("Black, White, or random? b/w/default=random").lower()

if inp in colourDict:
    colour = colourDict[inp]
else:
    c_index = randint(0,1)
    colour = colour_list[c_index]
    
# Initialise player objects and board object
player = Player(colour)
cpu = Player(colour_list[not(c_index)])
board = Board()
board.show()


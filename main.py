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
        self.home = 0
    
    def place(self):
        """Lets the player place a checker
        """
        self.placed += 1
        self.unplaced -= 1
        
    def hit(self):
        """Removes a player's checker from the board
        """
        self.placed -= 1
        self.unplaced += 1

    def send_home(self):
        """Lets a player send a checker home
        """
        self.placed -= 1
        self.home += 1
        
class Board:
    def __init__(self):
        self.board = [
                    [[[],[],[],[],[],[]],[[],[],[],[],[],[]]],
                    [[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
                    ]
        
    def updateBoard(self, dest, src, colour, opp):
        """Updates the contents of the board after a valid move

        Args:
            dest (list): The location of the desired move
            src (list): The location of the checker before the move
            colour (string): The checker's colour
            opp (Player): The opposition player's object
        """
        row, segment, col = dest[0],dest[1], dest[2]
        if self.board[row][segment][col][0] == colour:
            self.board[row][col].append(colour)
        else:
            self.board[row][col] = colour
            opp.hit()
        
        row, segment, col = src[0], src[1], src[2]
        self.board[row][segment][col].pop()
        
    def show(self):
        """Prints the board
        """
        for i in range(0, len(self.board[0])):
            print(self.board[i][0], self.board[i][1])


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


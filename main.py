import ai
import turn
import win
from random import randint

class Player:
    def __init__(self, colour):
        self.colour = colour
        self.unplaced = 0
        self.placed = 15
        self.home = 0
    
    def place(self):
        self.placed += 1
        self.unplaced -= 1
        
    def hit(self):
        self.placed -= 1
        self.unplaced += 1

    def send_home(self):
        self.placed -= 1
        self.home += 1
        
class Board:
    def __init__(self):
        self.board = [
                    [[['w','w'],[],[],[],[],[]],[[],[],[],[],[],[]]],
                    [[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
                    ]
        
    def updateBoard(self, dest, src, colour, opp):
        row, segment, col = dest[0],dest[1], dest[2]
        if self.board[row][segment][col][0] == colour:
            self.board[row][col].append(colour)
        else:
            self.board[row][col] = colour
            opp.hit()
        

        row, segment, col = src[0], src[1], src[2]
        self.board[row][segment][col].pop()
        
    def show(self):
        for i in range(0, len(self.board[0])):
            print(self.board[i][0], self.board[i][1])



colourDict = {'b':"Black", 'w':'White'}
inp = input("Black, White, or random? b/w/default=random").lower()

if inp in colourDict:
    colour = colourDict[inp]
else:
    colour = ['Black','White'][randint(0,1)]
    
colour = 'White'
player = Player(colour)
print(player.colour[0].lower())
cpu = Player('Black')
board = Board()
# board.updateBoard([0,0,0],[0,0,1],player.colour[0].lower(),cpu)
# board.show()

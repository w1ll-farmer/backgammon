# -n means there are n black pieces, n means there are n white pieces
# item slot 24 and 25 for bar, 26 and 27 for beard off pieces
board = [-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0,0,0,0]

def must_enter(board, colour):
    if colour > 0 and board[25] > 0:
        return False
    elif colour < 0 and board[24] < 0:
        return False
    else:
        return True
    

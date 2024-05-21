from random import randint
def roll():
    """Simulates the rolling of 2 dice
    
    Returns:
        int: The value of the number rolled on dice 1
        int: The value of the number rolled on dice 2
    """
    return randint(1, 6), randint(1, 6)

def isValidMove(board, dest, colour, src):
    """Checks a move is valid

    Args:
        board (Board.board): The 3D-list representing the board
        dest (list): 2D-list representing location of desired move
        colour (str): W or B representing black or white
        src (list): 2D-list representing position of checker before move
    Returns:
        _type_: _description_
    """
    src_row, src_segment, src_col = src[0],src[1], src[2]
    if len(board.board[src_row][src_segment][src_col]) == 0:
        return False
    # IMPLEMENT NO MOVING BACKWARDS
    dest_row, dest_segment, dest_col = dest[0],dest[1], dest[2]
    if len(board.board[dest_row][dest_segment][dest_col]) < 2:
        return True
    else:
        if board.board[dest_row][dest_segment][dest_col][0] == colour:
            return True
        else:
            return False
from turn import *

def check_inverted(current_board, boards, player):
    player *= -1
    inv_board = invert_board(current_board)
    inv_board_afters = []
    for board in boards:
        inv_board_afters.append(invert_board(board))
        
    return inv_board, inv_board_afters, player

def invert_board(current_board):    
    inv_board = current_board.copy()
    for i in range(len(current_board)):
        inv_board[i] = current_board[i] * -1
    temp_bar = inv_board[24]
    inv_board[24] = inv_board[25]
    inv_board[25] = temp_bar
    
    temp_home = inv_board[26]
    inv_board[26] = inv_board[27]
    inv_board[27] = temp_home
    return inv_board[0:24][::-1] + inv_board[24:]

def check_moves(board, boards, player, roll):
    inv_moves, inv_boards = get_valid_moves(-player, invert_board(board), roll)
    inv_boards = [invert_board(i) for i in inv_boards]
    if len(boards) == len(inv_boards):
        for i in boards:
            if i not in inv_boards:
                print(i)
                print("Not in inverse")
                exit()
        for j in inv_boards:
            if j not in boards:
                print(j)
                print("Not in forward board")
                exit()
    else:
        print("Different number of moves being shown")
        missing = [i for i in boards if i not in inv_boards] + [j for j in inv_boards if j not in boards]
        print(missing)
        print(len(boards), len(inv_boards))
        exit()
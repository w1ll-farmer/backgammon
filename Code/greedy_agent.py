from turn import get_home_info
# from turn import make_board
# from turn import update_board
# from turn import print_board
def evaluate(move, board_before, board_after, player):
    score = 0
    for m in move:
        start, end = m
        if end == 26 or end == 27:
            # Prioritise bearing off
            score += 4
            
        elif board_before[end] == -player:
            # Next hitting off blots
            score += 3
            print(abs(board_after[end]))
            if abs(board_after[end]) >= 2:
                score += 0.25
            
        elif board_before[end] == player:
            if abs(board_before[start]) > 2:
                # Making 2 walls when were 1
                score += 2.5
            elif board_before[start] == player:
                # Removing blot
                score += 2
            else:
                # Pushing wall forward
                score += 1.5
                
        elif end in get_home_info(player)[0]:
            # Player moves into their home board
            score += 1
            
        if board_after[start] == player:
            # Player doesn't generate advantage and leaves a blot
            score -= 1
    return score


# board_before = [0,0,-1,-1,0,3,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# move1, move2  = (5, 3), (7, 3)
# board_after = update_board(board_before, move1)
# board_after = update_board(board_after, move2)
# print(evaluate([move1, move2], board_before, board_after, 1))
# print_board(board_after)
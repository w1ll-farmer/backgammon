# from turn import get_home_info
# from turn import make_board
# from turn import update_board
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
            
        elif board_after[start] == player:
            # Player doesn't generate advantage and leaves a blot
            score -= 1
    return score


# board_before = make_board()
# board_after = update_board(make_board(), (7, 3))
# board_after = update_board(make_board(), (5, 3))
# print(evaluate([(7, 2), (5, 2)], board_before, board_after, 1))
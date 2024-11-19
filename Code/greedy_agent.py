from turn import get_home_info
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
                
        elif end in get_home_info(player, board_after)[0]:
            # Player moves into their home board
            score += 1
            
        if board_after[start] == player:
            # Player doesn't generate advantage and leaves a blot
            score -= 1
    return score

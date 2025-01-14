from turn import get_home_info
from turn import all_past

def count_blots(board, player):
    """Counts the number of a player's blots in a region

    Args:
        board (list(int)): Specified board region
        player (int): Whether player is controlling black or white

    Returns:
        int: Number of blots in specified region
    """
    return len([i for i in board if i == player])

def count_walls(board, player):
    """Counts the number of walls in a specified region

    Args:
        board (list(int)): Specified region checking in
        player (int): Whether player is controlling black or white

    Returns:
        int: Numbers of player's walls in region
    """
    if player == 1:
        return len([i for i in board if i > player])
    else:
        return len([i for i in board if i < player])

def eevaluate(move, board_before, board_after, player):
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

def evaluate(board_before, board_after, player):
    """Gives a score to a move

    Args:
        move (_type_): _description_
        board_before (list(int)): The board at the start of the turn
        board_after (list(int)): The resulting board if the move is made
        player (int): Whether player is controlling white or black

    Returns:
        int: The score associated to teh move
    """
    # Home information
    home_after = get_home_info(-player, board_after)[1]
    home_before = get_home_info(-player, board_before)[1]
    # Wall information
    walls_after = count_walls(board_after, player)
    walls_before = count_walls(board_before, player)
    wall_diff = walls_after - walls_before
    # Blot information
    player_blots_before = count_blots(board_before, player)
    player_blots_after = count_blots(board_after, player)
    blot_diff = player_blots_before - player_blots_after
    
    points = 0
    borne_off = 0
    
    # If home board is completely walled off
    if len([point for point in home_after if point*player > 1]) == 6:
        points += 17
        # Count how many opponent blots were hit in home
        num_home_opp_blots = len([i for i in home_before if player*i<0])
        if num_home_opp_blots > 0:
            points += num_home_opp_blots + 5
            
        # Check if piece(s) borne off
        if board_before[26] - board_after[26] > 0 or board_after[27] - board_before[27] > 0:
            borne_off = (board_before[26] - board_after[26]) + (board_after[27] - board_before[27])
        
        # Count num piece borne off and add to points
            if player == 1 and board_after[24] < 0:
                borne_off += 1
            elif player == -1 and board_after[25] > 0:
                borne_off += 1
        points += borne_off
    
    else:
        # Bearing off pieces
        if board_after[26] < board_before[26] or board_after[27] > board_before[27]:
            
            points += 13
            if not all_past(board_after):
                points -= sum([i for i in home_after if i == player])
                
                # Calc numbers of home hits
                num_home_opp_blots_before = count_blots(home_before, -player)
                num_home_opp_blots_after = count_blots(home_after, -player)
                points += (num_home_opp_blots_before - num_home_opp_blots_after)
        
        else:
            # If piece(s) hit off
            if board_after[24] < board_before[24] or board_after[25] > board_before[25]:
                if player_blots_before >= player_blots_after:
                    points += 11 + 0.5*(player_blots_before - player_blots_after)
                else:
                    # Left a piece exposed
                    points += 7

            elif not all_past(board_after):
                # Check if number of home walls have changed
                home_walls_after = count_walls(home_after, player)
                home_walls_before = count_walls(home_before, player)
                home_wall_diff = home_walls_after - home_walls_before
                
                if wall_diff > 0 and blot_diff > 0 and home_wall_diff > 0:
                    points += 9 + 0.2*wall_diff + 0.3*blot_diff + 0.1*home_wall_diff
                # Increase in number of walls
                elif wall_diff > 0:
                    points += 8 + 0.2*wall_diff
                # Decrease in number of blots
                elif blot_diff > 0:
                    points += 9 + 0.3*blot_diff
                # Increase in number of walls in home
                elif home_wall_diff > 0:
                    points += 7 + 0.1*home_wall_diff
                    
                # Made more blots
                if blot_diff < 0:
                    points += blot_diff
                # Made fewer walls
                if wall_diff < 0:
                    points += 0.8*wall_diff
            
            # Move all pieces past this turn  
            elif not all_past(board_before) and all_past(board_after):
                points += 6
                
            else:
                if blot_diff < 0:
                    points += blot_diff
                if wall_diff < 0:
                    points += 0.8*wall_diff
        
        if not all_past(board_after):
            # Maintain walls
            if wall_diff == 0:
                points += 0.09
            # No increase or decrease in blots
            if blot_diff == 0:
                points += 0.08
    return points


#### CHECK THIS ALL
"""
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-14,0,3,2,2,4,4,-1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-14,0,3,2,2,4,2,2,-1,0,0,0],1)) # 23
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-15,0,3,2,2,4,4,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-15,0,3,2,2,4,2,2,0,0,0,2],1)) # 19
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,-14,0,3,2,3,4,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-14,0,3,3,3,4,0,0,-1,0,0,1],1)) # 14
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,-14,0,3,2,3,4,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-14,0,3,3,3,4,0,-1,0,0,0,1],1)) # 13
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,-14,0,3,2,3,4,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-14,0,3,3,1,6,0,-1,0,0,0,1],1)) # 12
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,-14,0,3,1,3,4,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-14,0,3,4,0,6,0,2,-1,0,0,0],1)) # 11.59
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,-14,0,3,2,3,4,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-14,0,3,5,0,6,0,1,-1,0,0,0],1)) # 7
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,-14,0,3,2,3,4,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-14,0,3,4,0,6,0,2,-1,0,0,0],1)) # 11.08
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,-14,0,3,2,3,3,-1,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,-14,0,3,2,3,3,-1,2,0,0,0,0],1)) # 9.6
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,-14,0,3,2,3,3,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-14,0,5,2,3,3,-1,2,0,0,0,0],1)) # 9.5
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-14,1,-1,14,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-14,0,-1,14,1,0,0,0,0,0,0,0],1)) # 6
# print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,-14,0,-1,13,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-14,1,-1,13,0,0,0,0,0,0,0,0],1)) # 6
"""
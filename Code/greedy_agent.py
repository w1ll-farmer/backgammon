from turn import get_home_info
from turn import all_past

def count_blots(board, player):
    return len([i for i in board if i == player])

def count_walls(board, player):
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

def evaluate(move, board_before, board_after, player):
    
    home_after = get_home_info(-player, board_after)[1]
    home_before = get_home_info(-player, board_before)[1]
    
    walls_after = count_walls(board_after, player)
    walls_before = count_walls(board_before, player)
    wall_diff = walls_before - walls_after
    
    player_blots_before = count_blots(board_before, player)
    player_blots_after = count_blots(board_after, player)
    blot_diff = player_blots_after - player_blots_before
    
    points = 0
    borne_off = 0
    
    # If home board is completely walled off
    if len([point for point in home_after if point*player > 1]) == 6:
        points += 17
        # count number of blots in home before to see how many were hit
        # and replace with walls
        num_home_opp_blots = len([i for i in home_before if player*i<0])
        if num_home_opp_blots > 0:
            points += num_home_opp_blots + 5
        
        # Count num piece borne off and add to points
        if player == 1 and board_after[24] < 0:
            borne_off = board_after[27] - board_before[27]
        elif player == -1 and board_after[25] > 0:
            borne_off = board_before[26] - board_after[26]
        points += borne_off
    
    else:
        if board_after[26] < board_before[26] or board_after[27] > board_before[27]:
            points += 13
            if not all_past(board_after):
                points -= sum([i for i in home_after if i == player])
                
                # Calc numbers of home hits
                num_home_opp_blots_before = count_blots(home_before, -player)
                num_home_opp_blots_after = count_blots(home_after, -player)
                points += num_home_opp_blots_after - num_home_opp_blots_before
        
        else:
            if board_after[24] < board_before[24] or board_after[25] > board_before[25]:
                if player_blots_before >= player_blots_after:
                    points += 11 + 0.5*(player_blots_before - player_blots_after)
                else:
                    points += 7

            elif not all_past(board_after):
                home_walls_after = count_walls(home_after, player)
                home_walls_before = count_walls(home_before, player)
                home_wall_diff = home_walls_after - home_walls_before
                if wall_diff > 0 and blot_diff > 0 and home_wall_diff > 0:
                    points += 9 + 0.2*wall_diff + 0.3*blot_diff + 0.1*home_wall_diff
                    
                elif wall_diff > 0:
                    points += 8 + 0.2*wall_diff
                elif blot_diff > 0:
                    points += 9 + 0.3*blot_diff
            
                elif home_wall_diff > 0:
                    points += 7 + 0.1*home_wall_diff
                
                
            elif not all_past(board_before) and all_past(board_after):
                points += 6
                
            else:
                if blot_diff < 0:
                    points += blot_diff
                
                if wall_diff < 0:
                    points += 0.8*wall_diff
                    
        
        if not all_past(board_after):
            if wall_diff == 0:
                points += 0.09
            
            if blot_diff == 0:
                points += 0.08
    
    return points


#### CHECK THIS ALL
                
            
                
            
            
                
                
                
                
                
                
                
                
        
        
                
                
        
            
        
        
        
        
        
        
        
        

    
    return True

print(evaluate([1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,-1,-1,-1,0,0,0,0],[0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,-1,2,2,2,2,2,0,0,0,0],1))
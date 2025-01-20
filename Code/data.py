from turn import get_valid_moves, print_board, make_board
def log(board_before, roll,  move, board_after, player):
    myFile = open("./Data/log.txt",'a')
    myFile.write(f"{board_before}\t{roll}\t{move}\t{board_after}\t{player}\n")
    myFile.close()

def greedy_summarise():
    myFile = open("./Data/greedydata.txt")
    white_score = 0
    black_score = 0
    times_won = 0
    games_played = 0
    for line in myFile:
        line = line.strip("(")
        line = line.strip(")")
        white_points, black_points = line.split(",")
        black_points = int(black_points.strip('\n'))
        white_points = int(white_points)
        if white_points - black_points > 0:
            times_won +=1
        white_score += white_points
        black_score += 1
        games_played += 1
    return white_score, black_score, games_played, times_won

def timestep(timestep):
    myFile = open("./Data/randomvrandom.txt",'a')
    myFile.write(f"{timestep}\n")
    myFile.close()
    
def first_turn(p1):
    myFile = open("./Data/first-turn.txt","a")
    myFile.write(f"{p1}\n")
    myFile.close()
    
def calc_first():
    myFile = open("./Data/first-turn.txt","r")
    total = 0
    for line in myFile:
        total += int(line.strip("\n"))
    myFile.close()
    return total

def write_eval(eval, player):
    myFile = open("./Data/evaluations.txt","a")
    myFile.write(f"{eval, player}\n")
    myFile.close()
    
def calc_av_eval():
    whiteeval = 0
    blackeval = 0
    whiteturns = 0
    blackturns = 0
    white_pos, black_pos = 0,0
    black_moves = dict()
    white_moves = dict()
    myFile = open("./Data/evaluations.txt","r")
    for line in myFile:
        val, player = line.split(",")
        val = float(val[1:])
        if int(player.strip("\n")[:-1]) == 1:
            if val > 0: white_pos +=1
            whiteeval += val
            whiteturns += 1
            if val not in white_moves:
                white_moves[val] = 1
            else:
                white_moves[val] += 1
        else:
            if val > 0: black_pos +=1
            blackeval += val
            blackturns += 1
            if val not in black_moves:
                black_moves[val] = 1
            else:
                black_moves[val] += 1
    print(black_moves,"\n")
    print(white_moves)
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Convert the dictionaries to lists of evaluation scores and frequencies
    black_eval_scores = list(black_moves.keys())
    black_frequencies = list(black_moves.values())
    white_eval_scores = list(white_moves.keys())
    white_frequencies = list(white_moves.values())

    # Create a dataframe for seaborn
    import pandas as pd
    data = pd.DataFrame({
        "Evaluation Score": black_eval_scores + white_eval_scores,
        "Frequency": black_frequencies + white_frequencies,
        "Player": ["Black"] * len(black_eval_scores) + ["White"] * len(white_eval_scores)
    })

    # Plot the data using seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x="Evaluation Score", y="Frequency", hue="Player", palette="muted")
    plt.title("Frequency of Moves by Evaluation Score (Black vs White Players)")
    plt.xlabel("Evaluation Score")
    plt.ylabel("Frequency")
    plt.legend(title="Player")
    plt.show()



    print(white_pos, whiteturns-white_pos, black_pos, blackturns-black_pos)
    return whiteeval/whiteturns, blackeval/ blackturns
    
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

def save_roll(roll, player):
    myFile = open("./Data/roll.txt","a")
    myFile.write(f"{roll}, {player}\n")
    myFile.close()
    
def summarise_rolls():
    myFile = open("./Data/roll.txt","r")
    black_rolls = 0
    white_rolls = 0
    black_doubles = 0
    white_doubles = 0
    black_dist = 0
    white_dist = 0
    for line in myFile:
        line = line.strip("\n")
        print(line)
        roll1 = int(line[1])
        roll2 = int(line[4])
        player = int(line[-2:])
        if player > 0:
            if roll1 == roll2:
                white_doubles +=1
                white_dist += roll1 + roll2
            white_dist += roll1 + roll2
            white_rolls +=1
        else:
            if roll1 == roll2:
                black_doubles +=1
                black_dist += roll1 + roll2
            black_dist += roll1 + roll2
            black_rolls +=1
    print(white_rolls, white_doubles, white_dist)
    print(white_doubles/white_rolls)
    print(white_dist/white_rolls)
    print(black_rolls, black_doubles, black_dist)
    print(black_doubles/ black_rolls)
    print(black_dist/black_rolls)
    
    
def transform_moves(moves):
    transformed_moves = []
    
    for sublist in moves:
        transformed_sublist = []
        for start, end in sublist:
            # Apply the swapping rules for 24/25 and 26/27
            start = 25 if start == 24 else 24 if start == 25 else 27 if start == 26 else 26 if start == 27 else 23 - start
            end = 25 if end == 24 else 24 if end == 25 else 27 if end == 26 else 26 if end == 27 else 23 - end
            # Append the transformed tuple
            transformed_sublist.append((start, end))
        # Append the transformed sublist
        transformed_moves.append(transformed_sublist)
    
    return transformed_moves

def check_moves(board, moves, player, roll):
    inv_board = invert_board(board)
    inv_moves, inv_boards = get_valid_moves(-player, inv_board, roll)
    t_inv_moves = transform_moves(inv_moves)
    missing = [t_inv_moves[i] for i in range(len(moves)) if t_inv_moves[i] not in moves]
    if len(missing) > 0:
        print("ERROR DETECTED")
        print(f"Player {player} has {moves}")
        print(f"Player {-player} would have {t_inv_moves}")
        print(missing)
        exit()
        

# board = make_board()
# board[0] = -1
# board[1] = -1
# print(invert_board(board))
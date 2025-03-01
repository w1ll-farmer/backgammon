from turn import get_valid_moves, print_board, make_board
import os
def log(board_before, roll,  move, board_after, player):
    myFile = open(os.path.join("Data","log.txt"),'a')
    myFile.write(f"{board_before}\t{roll}\t{move}\t{board_after}\t{player}\n")
    myFile.close()

def greedy_summarise():
    myFile = open(os.path.join("Data","greedydata.txt"),'a')
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
    myFile = open(os.path.join("Data","randomvrandom.txt"),'a')
    myFile.write(f"{timestep}\n")
    myFile.close()
    
def first_turn(p1):
    myFile = open(os.path.join("Data","first-turn.txt"),'a')
    myFile.write(f"{p1}\n")
    myFile.close()
    
def calc_first():
    myFile = open(os.path.join("Data","first-turn.txt"),'r')
    total = 0
    for line in myFile:
        total += int(line.strip("\n"))
    myFile.close()
    return total

def write_eval(eval, player):
    myFile = open(os.path.join("Data","evaluations.txt"),'a')
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
    myFile = open(os.path.join("Data","evaluations.txt"),'r')
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
    # print(black_moves,"\n")
    # print(white_moves)
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # Convert the dictionaries to lists of evaluation scores and frequencies
    # black_eval_scores = list(black_moves.keys())
    # black_frequencies = list(black_moves.values())
    # white_eval_scores = list(white_moves.keys())
    # white_frequencies = list(white_moves.values())
    # ratio_dict = dict()
    # for key in black_eval_scores:
    #     if key in white_eval_scores:
    #         ratio_dict[key] = black_moves[key] / white_moves[key]
    # # Example dictionary
    # # Sorting by values in ascending order
    # sorted_dict = dict(sorted(ratio_dict.items(), key=lambda item: item[1]))
    # print(sorted_dict)
    # sorted_dict = dict(sorted(ratio_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_dict)
    # # Output: {'b': 1, 'c': 2, 'a': 3}

    # # Create a dataframe for seaborn
    # import pandas as pd
    # data = pd.DataFrame({
    #     "Evaluation Score": black_eval_scores + white_eval_scores,
    #     "Frequency": black_frequencies + white_frequencies,
    #     "Player": ["Black"] * len(black_eval_scores) + ["White"] * len(white_eval_scores)
    # })

    # # Plot the data using seaborn
    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=data, x="Evaluation Score", y="Frequency", hue="Player", palette="muted")
    # plt.title("Frequency of Moves by Evaluation Score (Black vs White Players)")
    # plt.xlabel("Evaluation Score")
    # plt.ylabel("Frequency")
    # plt.legend(title="Player")
    # plt.show()



    # print(white_pos, whiteturns-white_pos, black_pos, blackturns-black_pos)
    return whiteeval/whiteturns, blackeval/ blackturns
    


def save_roll(roll, player):
    myFile = open(os.path.join("Data","roll.txt"),'a')
    myFile.write(f"{roll}, {player}\n")
    myFile.close()
    
def summarise_rolls():
    myFile = open(os.path.join("Data","roll.txt"),'r')
    black_rolls = 0
    white_rolls = 0
    black_doubles = 0
    white_doubles = 0
    black_dist = 0
    white_dist = 0
    lastplayer = 0
    for line in myFile:
        line = line.strip("\n")
        roll1 = int(line[1])
        roll2 = int(line[4])
        player = int(line[-2:])
        count = 1
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
        if lastplayer == player:
            count += 1
            if count >= 3:
                print(f"Player {player} too many in a row")
        else:
            count == 1
        lastplayer = player
    print(white_rolls, white_doubles, white_dist)
    print(white_doubles)
    print(white_dist/white_rolls)
    print(black_rolls, black_doubles, black_dist)
    print(black_doubles)
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

def compare_eval_equity(evaluation, equity):
    myFile = open(os.path.join("Data","evalequitycompare.txt"),'a')
    myFile.write(f"{evaluation}, {equity}\n")
    myFile.close()
    
def get_eval_equity():
    myFile = open(os.path.join("Data","evalequitycompare.txt"),'r')
    evaluation = []
    equity = []
    for line in myFile:
        ev, eq = line.split(",")
        evaluation.append(float(ev))
        equity.append(float(eq))
    return evaluation, equity


def linear_regression(evaluation, equity):
    """
    Computes the linear regression coefficients (slope and intercept) 
    between evaluation and equity using the least squares method.

    Parameters:
      - evaluation: List of evaluation values (independent variable, X)
      - equity: List of equity values (dependent variable, Y)

    Returns:
      - slope (m)
      - intercept (b)
    """
        
    if len(evaluation) != len(equity) or len(evaluation) == 0:
        raise ValueError("Input lists must be of the same non-zero length.")

    n = len(evaluation)
    sum_x = sum(evaluation)
    sum_y = sum(equity)
    sum_xy = sum(x * y for x, y in zip(evaluation, equity))
    sum_x2 = sum(x ** 2 for x in evaluation)

    # Compute slope (m) and intercept (b)
    denominator = (n * sum_x2 - sum_x ** 2)
    if denominator == 0:
        raise ValueError("Denominator is zero, cannot compute regression.")

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    return slope, intercept
        
def r_squared(evaluation, equity, slope, intercept):
    """
    Computes the R² (coefficient of determination) for a linear regression model.
    
    Parameters:
      - evaluation: List of independent variable values (X)
      - equity: List of dependent variable values (Y)
      - slope: Computed slope from regression
      - intercept: Computed intercept from regression

    Returns:
      - R² value
    """
    mean_y = sum(equity) / len(equity)
    ss_total = sum((y - mean_y) ** 2 for y in equity)
    ss_residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(evaluation, equity))
    
    return 1 - (ss_residual / ss_total) if ss_total != 0 else 0

import numpy as np
from scipy.stats import norm

def map_equity_to_normal(evaluation, equity):
    """
    Maps equity values to a normal distribution of evaluation values.
    
    Parameters:
      - evaluation: List of independent variable values (X)
      - equity: List of dependent variable values (Y)

    Returns:
      - Mapped equity values following a normal distribution of evaluation values
    """
    mean_eval = np.mean(evaluation)
    std_eval = np.std(evaluation)

    # Normalize equity values to the standard normal distribution
    equity_z = (equity - np.mean(equity)) / np.std(equity)

    # Map to evaluation's normal distribution
    mapped_equity = mean_eval + equity_z * std_eval

    return mapped_equity

def check_correlation():
    evaluation, equity = get_eval_equity()
    slope, intercept = linear_regression(evaluation, equity)
    print(f"Equity = Evaluation*{slope}+{intercept}")
    print(f"R^2={r_squared(evaluation, equity, slope, intercept)}")
    print(np.mean(evaluation), np.std(evaluation), np.mean(equity), np.std(equity))
    print(max(equity))

def write_equity(equity, equitytype):
    myFile = open(os.path.join("Data",f"{equitytype}.txt"),'a')
    myFile.write(f"{equity}\n")
    myFile.close()
    
def normalise_equity():
    myFile = open(os.path.join("Data",f"WinnerEquity.txt"),'r')
    equity = []
    for line in myFile:
        eq = line.strip("\n")
        equity.append(float(eq))
    print("Advanced",np.mean(equity), np.std(equity))
    myFile = open(os.path.join("Data",f"LoserEquity.txt"),'r')
    equity = []
    for line in myFile:
        eq = line.strip("\n")
        equity.append(float(eq))
    print("Basic",np.mean(equity), np.std(equity))

def should_have_doubled():
    myFile = open(os.path.join("Data",f"AdvancedEquity.txt"),'r')
    equity = []
    for line in myFile:
        eq = line.strip("\n")
        equity.append(float(eq))
    print(len([eq for eq in equity if eq > 2.8575])/ len(equity))
    myFile = open(os.path.join("Data",f"BasicEquity.txt"),'r')
    equity = []
    for line in myFile:
        eq = line.strip("\n")
        equity.append(float(eq))
    print(len([eq for eq in equity if eq > 0.8198])/ len(equity))

def get_best_double_points():
    myFile = open(os.path.join("Data","adaptivevsgeneticdoubleon.txt"),'r')
    row_content = []
    for line in myFile:
        row_content.append(line.strip("\n"))
    tot = 0
    points = []
    drops = []
    tots = []
    count = 0
    for row in row_content[642:]:
        w_score, b_score, double_point, double_drop = row.split(",")
        w_score = int(w_score)
        b_score = int(b_score)
        tot += (w_score-b_score)
        if count % 4 == 0:
            double_point = float(double_point)
            double_drop = float(double_drop)
        if count % 4 == 3:
            points.append(double_point)
            drops.append(double_drop)
            tots.append(tot)
            tot = 0
        count += 1
    
    sorted_triplets = sorted(zip(tots, points, drops), key=lambda x: x[0], reverse=True)
    sorted_tots, sorted_points, sorted_drops = zip(*sorted_triplets)
    return sorted_triplets[:10]

def write_board_points(boards, score):
    for board in boards:
        myFile = open(os.path.join("Data","board_success_2.txt"),"a")
        myFile.write(f"{board}, {score}\n")
        myFile.close()

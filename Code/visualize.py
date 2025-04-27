import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt
import os
# from data import *



# total_white_score, total_black_score, games_played, times_won = greedy_summarise()


def head_to_head(myFile):
    white_score = []
    black_score = []
    white_wins = []
    black_wins = []
    for line in myFile:
        line = line.strip("(")
        line = line.strip(")")
        line = line.strip('\n')
        line = line.strip(")")
        white_points, black_points = line.split(",")
        black_points = int(black_points)
        white_points = int(white_points)
        if len(white_score) > 0:
            white_score.append(white_points+white_score[-1])
            black_score.append(black_points+black_score[-1])
        else:
            white_score.append(white_points)
            black_score.append(black_points)
        if white_points > black_points:
            if len(white_wins) > 0:
                white_wins.append(white_wins[-1]+1)
                black_wins.append(black_wins[-1])
            else:
                white_wins.append(1)
                black_wins.append(0)
        else:
            if len(white_wins) > 0:
                white_wins.append(white_wins[-1])
                black_wins.append(black_wins[-1]+1)
            else:
                white_wins.append(0)
                black_wins.append(1)
            


    games = list(range(1, len(white_score) + 1))  # Game numbers (1-based)

    print(white_wins[-1], black_wins[-1])
    print(white_score[-1], black_score[-1])
    # Create scatter plot

    plt.figure(figsize=(8, 6))
    plt.plot(games, white_score, color='blue', label='Adaptive Score', linewidth=3)
    plt.plot(games, black_score, color='red', label='Deep Score', linewidth=3)
    plt.plot(games, white_wins, color='purple', label='Adaptive Wins', linewidth=3)
    plt.plot(games, black_wins, color='orange', label='Deep Wins', linewidth=3)
    # Add labels, legend, and title
    plt.xlabel('Game Number')
    plt.ylabel('Cumulative Totals')
    plt.title('Performance Across 100 Cubeful Matches with Swapped Doubling Strategies')
    plt.legend()
    plt.grid(True)
    plt.show()



import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def plot_scores(myFile):
    episodes = []
    scores = []
    for line in myFile:
        line = line.strip("(")
        line = line.strip(")")
        line = line.strip('\n')
        line = line.strip(")")
        episode, score = line.split(",")
        episodes.append(int(episode))
        scores.append(float(score)/200)
    # episodes = np.array([i * interval for i in range(1, len(scores) + 1)])
    scores = np.array(scores)
    
    # Compute regression line
    # slope, intercept, _, _, _ = linregress(episodes, scores)
    # regression_line = slope * episodes + intercept
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores, marker='o', linestyle='-', color='b', label='Model Score')
    # plt.plot(episodes, regression_line, linestyle='--', color='r', label='Regression Line')
    
    plt.xlabel('Episodes')
    plt.ylabel('PPG Differential')
    plt.title('Model Performance vs GNUBG Beginner Mode')
    # plt.ylim(-100, 100)
    plt.legend()
    plt.grid(True)
    
    plt.show()

import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

def make_bar(txt_filename):
    game_groups = []  # List to store groups of games
    
    with open(txt_filename, newline='') as txtfile:
        reader = csv.reader(txtfile)
        rows = list(reader)  # Read all rows into a list
        
        for i in range(0, len(rows), 2):  # Process every pair of rows
            if i + 1 >= len(rows):
                break  # Ensure there's a corresponding cumulative score row
            
            # Extract win records (Odd row)
            wins1, wins2 = map(int, rows[i])
            
            # Extract cumulative scores (Even row)
            score1, score2 = map(int, rows[i + 1])
            
            # Store the data in the desired order: Score1, Score2, Win1, Win2
            game_groups.append([score1, score2, wins1, wins2])
    
    # Flatten grouped games while keeping track of gaps
    values = []
    positions = []
    colors = []
    spacing = 1  # Start x positions at 1
    color_palette = ['orange', 'green', 'red', 'purple', 'brown']#, 'pink', 'gray', 'olive', 'cyan']
    
    game_index = 0
    for game in game_groups:
        game_color = color_palette[game_index % len(color_palette)]  # Assign a different color per game pair
        values.append(game[0])
        positions.append(spacing)
        colors.append('blue')  # First score always blue
        spacing += 1
        
        values.append(game[1])
        positions.append(spacing)
        colors.append(game_color)  # Different color per game pair
        spacing += 1
        
        values.append(game[2])
        positions.append(spacing)
        colors.append('blue')  # First win always blue
        spacing += 1
        
        values.append(game[3])
        positions.append(spacing)
        colors.append(game_color)  # Different color per game pair
        spacing += 1.8  # Add extra space between different games
        
        game_index += 1
    
    # Plot bar chart with gaps between groups
    plt.figure(figsize=(10, 5))
    bars = plt.bar(positions, values, color=colors, edgecolor='black')
    
    # Annotate bars with values
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(value), 
                 ha='center', va='bottom', fontsize=12)
    
    # Add the legend
    # Create custom legends for each color used
    legend_labels = {
        'blue': 'ReinforceV2',
        'orange': 'Deep',
        'green': 'Adaptive',
        'red': 'Genetic',
        'purple': 'Expectimax',
        'brown': 'Greedy',
        # 'pink': 'Score Win Pair (Pink)',
        # 'gray': 'Score Win Pair (Gray)',
        # 'olive': 'Score Win Pair (Olive)',
        # 'cyan': 'Score Win Pair (Cyan)'
    }
    
    handles = [mlines.Line2D([0], [0], color=color, lw=4, label=legend_labels.get(color, color)) for color in color_palette]
    plt.legend(handles=handles, loc=10)
    
    plt.xlabel("Measure of Performance")
    plt.ylabel("Values")
    plt.title("Results from the Reinforcement Agent Playing 50 Cubeless First-to-25 Matches")
    plt.xticks(positions, ["Score", "Score", "Wins", "Wins"]*(len(positions)//4), rotation=20)  # Repeating x-tick labels
    plt.show()



    

# myFile = open(os.path.join("Data","cubelessRLvGREEDY.txt"))
# head_to_head(myFile)
myFile = open(os.path.join("Data","RL","benchmark3RL2.txt"))
plot_scores(myFile)
# myFile = os.path.join("Data","Results","RL2agents.txt")
# make_bar(myFile)

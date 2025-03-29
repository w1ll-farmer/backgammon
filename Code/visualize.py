import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt
import os
# from data import *



# total_white_score, total_black_score, games_played, times_won = greedy_summarise()

myFile = open(os.path.join("Data","RL","beginnervRLstart115k.txt"))
white_score = []
black_score = []
white_wins = []
black_wins = []
for line in myFile:
    line = line.strip("(")
    line = line.strip(")")
    line = line.strip('\n')
    line = line.strip(")")
    # white_points, black_points = line.split(",")
    # black_points = int(black_points)
    # white_points = int(white_points)
    white_score.append(int(line))
    # if len(white_score) > 0:
    #     white_score.append(white_points+white_score[-1])
    #     black_score.append(black_points+black_score[-1])
    # else:
    #     white_score.append(white_points)
    #     black_score.append(black_points)
    # if white_points > black_points:
    #     if len(white_wins) > 0:
    #         white_wins.append(white_wins[-1]+1)
    #         black_wins.append(black_wins[-1])
    #     else:
    #         white_wins.append(1)
    #         black_wins.append(0)
    # else:
    #     if len(white_wins) > 0:
    #         white_wins.append(white_wins[-1])
    #         black_wins.append(black_wins[-1]+1)
    #     else:
    #         white_wins.append(0)
    #         black_wins.append(1)
        


# games = list(range(1, len(white_score) + 1))  # Game numbers (1-based)

# print(white_wins[-1]), black_wins[-1])
# print(white_score[-1], black_score[-1])
# Create scatter plot

# plt.figure(figsize=(8, 6))
# plt.plot(games, white_score, color='blue', label='Adaptive Score', linewidth=3)
# # plt.plot(games, black_score, color='red', label='Genetic Score', linewidth=3)
# # plt.plot(games, white_wins, color='purple', label='Adaptive Wins', linewidth=3)
# # plt.plot(games, black_wins, color='orange', label='Genetic Wins', linewidth=3)
# # Add labels, legend, and title
# plt.xlabel('Game Number')
# plt.ylabel('Cumulative Totals')
# plt.title('Cumulative Scores and Wins Across 100 Cubeless First-to-25 Matches')
# plt.legend()
# plt.grid(True)
# plt.show()



import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def plot_scores(scores, interval=5000):
    episodes = np.array([i * interval for i in range(1, len(scores) + 1)])
    scores = np.array(scores)
    
    # Compute regression line
    slope, intercept, _, _, _ = linregress(episodes, scores)
    regression_line = slope * episodes + intercept
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores, marker='o', linestyle='-', color='b', label='Model Score')
    plt.plot(episodes, regression_line, linestyle='--', color='r', label='Regression Line')
    
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Model Performance Over Episodes')
    plt.ylim(-100, 100)
    plt.legend()
    plt.grid(True)
    
    plt.show()


# Example usage
scores = white_score  # Replace with your actual scores
plot_scores(scores)

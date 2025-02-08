import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt
# from data import *



# total_white_score, total_black_score, games_played, times_won = greedy_summarise()

myFile = open("./Data/greedydata.txt")
white_score = []
black_score = []
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

games = list(range(1, len(white_score) + 1))  # Game numbers (1-based)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.plot(games, white_score, color='blue', label='Greedy Agent 1', linewidth=3)
plt.plot(games, black_score, color='red', label='Greedy Agent -1', linewidth=3)

# Add labels, legend, and title
plt.xlabel('Game Number')
plt.ylabel('Cumulative Score')
plt.title('Cumulative Scores Across 2000 Games')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt
# from data import *
lst = [[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]]

# Convert lists to tuples, use set for uniqueness, then convert back to lists
unique_lists = list(map(list, set(map(tuple, lst))))

print(unique_lists)


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
plt.plot(games, white_score, color='blue', label='Greedy Agent', linewidth=3)
plt.plot(games, black_score, color='red', label='Random Agent', linewidth=3)

# Add labels, legend, and title
plt.xlabel('Game Number')
plt.ylabel('Cumulative Score')
plt.title('Cumulative Scores across 98 First-to-Five Matches')
plt.legend()
plt.grid(True)
plt.show()

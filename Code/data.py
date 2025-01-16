# def log(board_before,roll,  move, board_after):
#     myFile = open("./Data/log.txt",'a')
#     myFile.write(f"{board_before}\t{roll}\t{move}\t{board_after}\n")
#     myFile.close()

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
    
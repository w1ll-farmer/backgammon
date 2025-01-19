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
    myFile = open("./Data/evaluations.txt","r")
    for line in myFile:
        val, player = line.split(",")
        val = float(val[1:])
        if int(player.strip("\n")[:-1]) == 1:
            if val > 0: white_pos +=1
            whiteeval += val
            whiteturns += 1
        else:
            if val > 0: black_pos +=1
            blackeval += val
            blackturns += 1
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
    
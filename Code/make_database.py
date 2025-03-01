from turn import *
from random import randint, uniform, gauss

def generate_random_board():
    white_remaining = 15
    black_remaining = 15    
    board = [0]*28
    while white_remaining > 0 or black_remaining > 0:
        pos = randint(0, 27)
        x = uniform(-black_remaining,white_remaining)
        player = -1 if x < 0 else 1
        if player == 1 and pos != 24 and pos != 26 and board[pos] > -1:
            point = int(gauss(2.5, 2)) if white_remaining >= 7 else randint(0, white_remaining)
            while point > white_remaining or point < 0:
                point = int(gauss(2.5, 2))
            board[pos] += point
            white_remaining -= point
        elif player == -1 and pos != 25 and pos != 27 and board[pos] < 1:
            point = int(gauss(2.5, 2)) if black_remaining >= 7 else randint(0,black_remaining)
            while point < 0 or point > black_remaining:
                point = int(gauss(2.5, 2))
            board[pos] -= point
            black_remaining -= point
            
    return board

def make_opening(make_stuff=True):
    board = make_board()
    generated = {}
    for roll1 in range(1, 7):
        for roll2 in range(roll1, 7):
            moves, boards = get_valid_moves(1, board, (roll1, roll2))
            if make_stuff:
                myFile = open("Data/Deep/Opening/train.txt",'a')
            for b in boards:
                m = moves[boards.index(b)]
                key = str(b)
                if key not in generated:
                    generated[key] = 1
                    for p in b:
                        if make_stuff:
                            myFile.write(f"{p},")
                    
                    if make_stuff:
                        myFile.write(f"{m}\n")
    if make_stuff: myFile.close()

def gap_fill():
    myFile = open("Data/Deep/Opening/train.txt",'r')
    for line in myFile:
        line.strip("\n")
        board, movescore = line.split("[")
        print(board)
        move, score = movescore.split("]")
        print(score)
        if len(score) == 1:
            score = ",-0.05\n"
        myNewFile = open("Data/Deep/Opening/train2.txt",'a')
        myNewFile.write(f"{board}{score[1:]}")
        myNewFile.close()
    myFile.close()
        
def successful_boards():
    myFile = open(os.path.join("Data","board_success_2.txt"),'r')
    times_appear = {}
    total_score = {}
    avg_score = {}
    for line in myFile:
        board, score = line.split("],")
        board = board.strip("[")
        score = int(score.strip("\n"))
        if board in times_appear:
            times_appear[board] +=1
            total_score[board] += score
        else:
            times_appear[board] = 1
            total_score[board] = score
    myFile.close()
    for board in times_appear:
        if times_appear[board] > 49:
            avg_score[board] = total_score[board] / times_appear[board]
    
    for board in avg_score:
        myFile = open(os.path.join("Data","Deep","BoardEquity",f"board_equity_db.txt"),'a')
        myFile.write(f"{board},{avg_score[board]}\n")
        myFile.close()
        
successful_boards()

        
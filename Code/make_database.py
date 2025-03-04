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
        # print(board)
        move, score = movescore.split("]")
        # print(score)
        if len(score) == 1:
            score = ",-0.05\n"
        myNewFile = open("Data/Deep/Opening/train2.txt",'a')
        myNewFile.write(f"{board}{score[1:]}")
        myNewFile.close()
    myFile.close()
        
def successful_boards():
    myFile = open(os.path.join("Data","board_success_5.txt"),'r')
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
        if times_appear[board] > 99:
            avg_score[board] = total_score[board] / times_appear[board]
    
    for board in avg_score:
        myFile = open(os.path.join("Data","Deep","BoardEquity",f"board_equity_db.txt"),'a')
        myFile.write(f"{board},{avg_score[board]}\n")
        myFile.close()

def use_prev_board():
    myFile = open(os.path.join("Data","board_success.txt"),'r')
    times_appear = {}
    for line in myFile:
        board, score = line.split("],")
        board = board.strip("[")
        board = board.strip()
        board2 = board.split(",")
        board2 = [int(i) for i in board2 if i != " "]
        if board in times_appear:
            times_appear[board] +=1
        else:
            times_appear[board] = 1
    myFile.close() 
    boards = []
    for board in times_appear:
        if times_appear[board] < 10:
            board = board.strip("[")
            board = board.strip()
            board2 = board.split(",")
            board2 = [int(i) for i in board2 if i != " "]
            boards.append(board2)
    # print(len(boards))
    return boards

def generate_cp_db():
    myFile = open(os.path.join("Data","board_success_known.txt"),'r')
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
        if times_appear[board] > 99:
            avg_score[board] = total_score[board]/times_appear[board]
    
    
    for b in avg_score:
        board = b.strip()
        board = b.split(",")
        board = [int(i) for i in board if i != " "]
        input_vector = []
        for i in range(24):
            point_encoding = convert_point(board[i])
            input_vector += point_encoding
        for i in range(24):
            input_vector.append(prob_opponent_can_hit(1, board, i))
        for i in range(24, 26):
            input_vector += convert_bar(board[i])
        _, home = get_home_info(1, board)
        _, opp_home = get_home_info(-1, board)
        # % home points occupied
        input_vector.append(len([i for i in home if i > 0])/6)
        # % opp home points occupied
        input_vector.append(len([i for i in opp_home if i > 0])/6)
        # % pieces in home
        input_vector.append(sum([i for i in home if i >0])/15)
        # Prime?
        input_vector.append(1 if calc_prime(board, 1) > 3 else 0)
        # pip count
        input_vector += decimal_to_binary(calc_pips(board, 1))
        input_vector += decimal_to_binary(calc_pips(board, -1))
        
        # chance blockade can't be passed
        input_vector.append(calc_blockade_pass_chance(board, 1))
        myFile = open(os.path.join("Data","Deep","289-input-x.txt"),"a")
        input_vector = str(input_vector)[1:-1]
        myFile.write(f"{input_vector}\n")
        myFile.close()
        
        myFile = open(os.path.join("Data","Deep","289-input-y.txt"),"a")
        myFile.write(f"{avg_score[b]}\n")
        myFile.close()

def convert_point(point):
    base = [0]* 10
    if point < 0:
        for i in range(0, 5):
            if point <= i-5 and (i == 0 or i == 3) or point == i-5:
                base[i] = 1
    elif point > 0:
        for i in range(5, 10):
            if point >= i - 4 and (i == 9 or i == 6) or point == i-4:
                base[i] = 1
    return base


def convert_bar(point):
    base = [0]*3
    if point < 0:
        for i in range(0, 2):
            if point <= i-2 and i ==0 or point == i-2:
                base[i]=1
    elif point > 0:
        for i in range(0, 3):
            if point >= i and i == 2 or point == i:
                base[i]=1
    return base


generate_cp_db()
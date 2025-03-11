from turn import *
from random import randint, uniform, gauss

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
    myFile = open(os.path.join("Data","board_success_online.txt"),'r')
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

# 534 rows in initial
# 1267 rows after board_success_2
# 1857 rows after board_success_4
# 2185 after board_success_5
# 2695 after board_success_online
# generate_cp_db()

def encode_cp(b):
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
    return input_vector
    
def comparison_paradigm(end, start = 0, testset=False):
    for i in range(start, end):
        myFile = open(os.path.join("Data","Deep","GNUBG-data",f"positions{i}.txt"),"r")
        raw_boards, equities = [], []
        for line in myFile:
            raw_board, equity = line.split("],")
            raw_boards.append(raw_board[1:])
            equities.append(float(equity.strip()))
        myFile.close()
        used_boards = []
        while len(used_boards) < len(raw_boards) -1:
            index = randint(0, len(raw_boards)-1)
            while index in used_boards:
                index = randint(0, len(raw_boards)-1)
            used_boards.append(index)
            board1 = raw_boards[index]
            equity1 = equities[index]
            new_index = randint(0, len(raw_boards)-1)
            while new_index in used_boards:
                new_index = randint(0, len(raw_boards)-1)
            used_boards.append(new_index)
            board2, equity2 = raw_boards[new_index], equities[new_index]
            if equity1 > equity2:
                Y = 1
                valY = 0
            elif equity1 < equity2:
                Y = 0
                valY = 1
            else:
                continue
            encoded_board1 = str(encode_cp(board1))
            encoded_board2 = str(encode_cp(board2))
            if testset == False:
                # Write to train set
                myFile = open(os.path.join("Data","Deep","BoardEquity","train.txt"),"a")
                myFile.write(f"{encoded_board1[1:-1]},{encoded_board2[1:-1]},{Y}\n")
                myFile.close()
                # Write to Validation Set
                # Same boards as train set but other way round, try to enforce symmetry
                myFile = open(os.path.join("Data","Deep","BoardEquity","validation.txt"),"a")
                myFile.write(f"{encoded_board2[1:-1]},{encoded_board1[1:-1]},{valY}\n")
                myFile.close()
            else:
                myFile = open(os.path.join("Data","Deep","BoardEquity","test.txt"),"a")
                myFile.write(f"{encoded_board1[1:-1]},{encoded_board2[1:-1]},{Y}\n")
                myFile.close()

if __name__ == "__main__":
    data_size = len(os.listdir(os.path.join("Data","Deep","GNUBG-data"))) - 1
    train_end =int(0.85*data_size)
    print("Train")
    comparison_paradigm(train_end, start = 0, testset=False)
    print("Test")
    comparison_paradigm(data_size, start = train_end, testset=True)
import subprocess
import os
import numpy as np
import re
from turn import *
from testfile import *
from make_database import *

def transform_board(board):
    """Maps the board from OpenBG to GNUBG format

    Args:
        board (list(int)): Board representation

    Returns:
        str: GNUBG board representation
    """
    
    gnuboard = [0]*26
    gnuboard[0] = board[25]
    gnuboard[25] = board[24]
    for i in range(1,25):
        gnuboard[i] = board[i-1] #24-i
    
    str_board = ""
    for i in gnuboard:
        str_board += f"{i} "
    
    return str_board.strip()

def get_move(board, roll, cube_val=1):
    """Calls GNUBG and returns output of hint
    
    Args:
        board (list(int)): Board representation
        roll (list(int)): The dice roll

    Returns:
        str: The raw output of GNUBG
    """
    
    # Code from https://github.com/nitzankoll/backGammon-analyzer/blob/main/backgammon.py 
    if roll is not None:
        commands = f"""
            new game
            set player 1 name white
            set player 0 name black
            set board simple {board}
            set turn 1
            set dice {roll[0]} {roll[1]}
            hint
            quit
            y"""
    else:
        commands = f"""
            new game
            set player 1 name white
            set player 0 name black
            set board simple {board}
            set turn 1
            set cube value {cube_val}
            hint
            quit
            y"""
    try:
        # print("Opening process...")
        # Start gnubg-cli process
        process = subprocess.Popen(
            ["gnubg","-t"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Communicate with gnubg-cli by sending commands
        # print("Communicating...")
        stdout, stderr = process.communicate(commands, timeout=60)
        # print("Terminating...")
        process.terminate()
        # print("Waiting...")
        process.wait()
        
        """
        # Debug output
        # print("GNU Backgammon Output:")
        # print(stdout)
        # print("Errors (if any):")
        # print(stderr)
        """
        return stdout
    except subprocess.TimeoutExpired:
        # print("Process timed out, starting new one")
        process.kill()
        return None
    except FileNotFoundError:
        # print("Error: gnubg-cli not found. Check the path.")
        return None

    return None

def hint_parser(gnu_output):
    """Returns the moves and their equities from gnubg output

    Args:
        gnu_output (str): The raw output from GNUBG hint

    Returns:
        list(tuple(str)): The moves and their associated equities
    """
    moves = []
    
    if "There are no legal moves" in gnu_output:
        return ["There are no legal moves"]
    
    # Regex pattern to capture all moves and their equities
    move_matches = re.findall(r"(\*?\s*\d+\.\s+Cubeful\s+\d-ply\s+([^\n]+)\s+Eq\.: ([^\n]+))", gnu_output)
    for match in move_matches:
        move_description = match[1].strip()  # The move itself
        equity = match[2].strip()  # The equity value
        moves.append((move_description, equity))
    
    return moves

def convert_to_opengammon(board, raw_moves, roll, play=False):
    """Converts moves and boards back to OpenBG format

    Args:
        raw_moves (list(tuple(str))): The raw moves and equity pairs
        roll (list(int)): The roll 

    Returns:
        list(tuple(int)), list(int), list(float): The moves, boards and equities
    """
    num_moves = len(raw_moves)
    moves, boards, equities = [], [], []
    for i in range(num_moves):
        move_string, eq_str = raw_moves[i]
        # print("Move before conversion:", move_string, roll)
        move = (convert_move(move_string, roll, board))
        moves.append(move)
        board_copy = board.copy()
        # print(board_copy)
        pos_moves, _ = get_valid_moves(1, board, roll)
        if move not in pos_moves:
            furthest_back = get_furthest_back(board, 1)
            if furthest_back == move[0][0]:
                move = [(furthest_back, furthest_back - min(roll)), (furthest_back - min(roll), 27)]
            if move not in pos_moves:
                print(f"{move} not possible")
                print(pos_moves)
                exit()
        for m in move:
            board_copy = update_board(board_copy, m)
        # print(board_copy, move)
        boards.append(board_copy)
        if "(" in eq_str:
            eq_str, diff = eq_str.split("(")
        equities.append(float(eq_str.strip()))
    # print("Converted to opengammon")
    return moves, boards, equities
        

def convert_move(moves, roll, board):
    """Converts move from GNUBG to OpenBG

    Args:
        moves (str): The GNUBG encoded move
        roll (list(int)): The dice roll

    Returns:
        List | Tuple: The OpenBG encoded move
    """
    moves = moves.split()
    open_format = []
    for move in moves:
        print("submove",move)
        if "off" in move:
            move = move.strip()
            if move[-1] == "f":
                move = move[:-3] + "28"
            else:
                index = move.index("o")
                move = move[:index] + "28"+ move[index+3:]
        elif "bar" in move:
            move = move.strip()
            move = "26"+move[3:]
        if move.count('/') > 1: # One piece being moved only, hits at least one on the way
            base = list_to_str([i for i in move if i != "*"], False,False)
            base = base.split("/")
            for i in range(len(base)-1):
                base_move = extract_base_move(f"{base[i]}/{base[i+1]}",roll, board)
                if isinstance(base_move,list):
                    for i in range(len(base_move)):
                        open_format.append(base_move[i])
                else:
                    open_format.append(base_move)
        elif "(" in move:
            base, repeat = move.split("(")
            repeat = int(repeat.strip(')'))
            print("base",base)
            if "*" in move:
                base = list_to_str([i for i in base if i != "*"], False, False)
                base_move = extract_base_move(base, roll, board)
                base_move*=2
                print("base_move",base_move)
                if isinstance(base_move, tuple):
                    open_format.append((base_move[0], base_move[1]))
                    open_format.append((base_move[2], base_move[3]))
                else:
                    open_format.append((base_move[0]))
            else:
                for r in range(repeat):
                    x=extract_base_move(base, roll, board)
                    print(x)
                    if isinstance(x, tuple):
                        open_format.append(x)
                    else:
                        open_format += x
        elif "*" in move:
            base = list_to_str([i for i in move if i != "*"], False, False)
            print(base)
            base_move = extract_base_move(base, roll, board)
            if isinstance(base_move,list):
                for i in range(len(base_move)):
                    open_format.append(base_move[i])
            else:
                open_format.append(base_move)
        else:
            base_move = extract_base_move(move, roll, board)
            if isinstance(base_move,list):
                for i in range(len(base_move)):
                    open_format.append(base_move[i])
            else:       
                open_format.append(base_move)
            # print(open_format)
            
    return open_format

def extract_base_move(base_move, roll, board):
    """Helper function for convert_move

    Args:
        base_move (str): The GNUBG encoded submove
        roll (list(int)): The dice roll

    Returns:
        List | Tuple: The OpenBG encoded submove
    """
    start, end = base_move.split("/")
    print(start, end)
    try:
        start, end = (int(start)-1, int(end)-1)
    # except ValueError:
    #     if end[-1] == "*":
    #         end = end[:-1]
    #     try:
    #         start, end = (int(start)-1, int(end)-1)
    except ValueError:
        print("Value error")
        print(base_move)
        exit()
            
    if (start - roll[0] > end and start - roll[1] > end) or \
        (end == 27 and start - roll[0] - roll[1] < 0 and (start - roll[0] >= 0 and start-roll[1] >= 0)):
        if not is_double(roll):
            if start == 25:
                s = 24
            else:
                s = start
            ret = [()]*2
            if board[start - roll[0]] < 0:
                ret[0] = (start, s - roll[1])
                ret[1] = (s - roll[1], end)
            else:
                ret[0] = (start, s - roll[0])
                ret[1] = (s - roll[0], end)
            print("ret", ret)
            return ret
        elif start == 25:
            s = 24
            diff = s - end
            moves = int(diff/roll[0])
            ret = [()]*moves
            for i in range(moves):
                ret[i] = (start - roll[0]*i, start-roll[0]*(i+1))
            print("ret", ret)
            return ret
        else:
            diff = start - end
            moves = int(diff/roll[0])
            ret = [()]*moves
            for i in range(moves):
                ret[i] = (start - roll[0]*i, start-roll[0]*(i+1))
            print("ret", ret)
            return ret
    print(start, end, "start end")
    return start, end
# print(extract_base_move("2/28", [4,2], [2, 12, 0, 1, 0, 0, 0, 0, 0, -5, 0, -2, 0, 0, -2, 0, 0, -4, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0]))

def get_move_equities(board, roll, player):
    if player == -1:
        board = invert_board(board)
    transformed_board = transform_board(board)
    # print("Getting moves...")
    gnu_output = get_move(transformed_board, roll)
    while gnu_output is None or is_double(roll):
        roll = roll_dice()
        gnu_output = get_move(transformed_board, roll)
    # print("Parsing input...")
    
    raw_moves = hint_parser(gnu_output)
    # print("Converting to opengammon...")
    return convert_to_opengammon(board, raw_moves, roll)

def write_move_equities(board, roll, player, i):
    moves, boards, equities = get_move_equities(board, roll, player)
    myFile = open(os.path.join("Data","Deep","GNUBG-data","Equity",f"positions{i-1}.txt"),"w")
    for i in range(len(moves)):
        myFile.write(f"{boards[i]},{equities[i]}\n")
    myFile.close()

def random_board_equities():
    positions = len(os.listdir(os.path.join("Data","Deep","GNUBG-data","Equity")))
    for i in range(100000):
        print(f"Number of boards generated: {i+positions}")
        board = generate_random_race_board()
        # board = make_board()
        # board[23] -= 1
        # board[25] += 1
        roll = roll_dice()
        while is_double(roll):
            roll = roll_dice()
        print("Dice:", roll)
        moves, _ = get_valid_moves(1, board, roll)
        while len(moves) == 0 or is_double(roll) or len(moves) > 50:
            roll = roll_dice()
            board = generate_random_race_board()
            moves, _ = get_valid_moves(1, board, roll)
        print(len(moves))
        write_move_equities(board, roll, 1, i+positions)

def extract_cube_action(gnu_output):
    action = re.search(r"Proper cube action:\s*([A-Za-z\s,]+)", gnu_output)
    
    action = action.group(1).strip()
    return action

def get_optimal_doubling(board, cube_val=1, player=1):
    if player == -1:
        board = invert_board(board)
    transformed_board = transform_board(board)
    gnu_output = get_move(transformed_board, None, cube_val)
    while gnu_output is None:
        gnu_output = get_move(transformed_board, None, cube_val)
    return extract_cube_action(gnu_output)

def write_optimal_doubling(board):
    action = get_optimal_doubling(board)
    inverted_board = invert_board(board)
    inverse_action = get_optimal_doubling(inverted_board)
    print(action, inverse_action)
    offer = 1 if action[0] == "D" else 0
    accept = 0 if action[-1] == "s" else 1
    
    inv_offer = 1 if inverse_action[0] == "D" else 0
    inv_accept = 0 if inverse_action[-1] == "s" else 1
    # Write offer decision
    path = os.path.join("Data","Deep","GNUBG-data","Cube")
    offerFile = open(os.path.join(path, "Offer",f"positions.txt"),"a")
    offerFile.write(f"{board},{offer}\n")
    offerFile.write(f"{inverted_board},{inv_offer}\n")
    offerFile.close()
    
    # Invert board so can be used to know when to drop/take double
    
    acceptFile = open(os.path.join(path, "Accept",f"positions.txt"),"a")
    acceptFile.write(f"{inverted_board},{accept}\n")
    acceptFile.write(f"{board},{inv_accept}\n")
    acceptFile.close()
    
def random_cube_decisions():
    i = 0
    while True:
        print(f"Generated {i} cube decisions")
        board = generate_random_board()
        write_optimal_doubling(board)
        i += 1

def move_parser(gnu_output):
    """Returns the moves and their equities from gnubg output

    Args:
        gnu_output (str): The raw output from GNUBG hint

    Returns:
        list(tuple(str)): The moves and their associated equities
    """
    moves = []
    
    if "There are no legal moves" in gnu_output:
        return ["There are no legal moves"]
    
    # Regex pattern to capture all moves and their equities
    move_matches = re.findall(r"(\*?\s*\d+\.\s+Cubeful\s+\d-ply\s+([^\n]+)\s+Eq\.: ([^\n]+))", gnu_output)
    return move_matches[0][1].strip()  

def invert_move(move):
    inverted_move = []
    for m in move:
        start, end = m
        inv_start, inv_end = 0,0
        if start == 25:
            inv_start = 24
        else:
            inv_start = 23-start
        # if end == 26:
        #     inv_end = 27
        if end == 27:
            inv_end = 26
        else:
            inv_end = 23-end
        inverted_move.append((inv_start, inv_end))
    return inverted_move
        
def gnubg_move(board, roll, player, valid_moves, valid_boards):
    valid_moves_copy = valid_moves.copy()
    if len(valid_boards) == 1:
        return valid_moves[0], valid_boards[0]
    if player == -1:
        board = invert_board(board)
        valid_moves_copy = [invert_move(m) for m in valid_moves]
    transformed_board = transform_board(board)
    gnu_output = get_move(transformed_board, roll)
    # print("get move", gnu_output)
    # gnu_output = get_first_only(gnu_output)
    # print("First only", gnu_output)
    # sanitised_move = sanitise_move(gnu_output)
    sanitised_move = hint_parser(gnu_output)[0][0]
    
    
    
    
    move = correct_moves(convert_move_notation(sanitised_move, roll))
    if player == -1:
        move = invert_move(move)
    for m in move:
       board = update_board(board, m) 
    if board in valid_boards:
        return move, board
    else:
        try:
            move = move[::-1]
            board = valid_boards[valid_moves.index(move)]
        except:
            
            print(board)
            
            exit()
    # return move, board

def gnubg_accept_double(board, cube_val, player):
    action = get_optimal_doubling(board, cube_val, player)
    accept = 0 if action[-1] == "s" else 1
    return bool(accept)

def gnubg_offer_double(board, cube_val, player):
    action = get_optimal_doubling(board, cube_val, player)
    offer = 1 if action[0] == "D" else 0
    return bool(offer)

def convert_move_notation(gnubg_moves, dice_rolls=None):
    # Updated regex to capture full move sequences, including hits (*) and repetitions (n)
    # Updated regex to correctly capture moves, including hits (*) and repetitions (n)
    move_pattern = re.compile(r'(bar|\d+)((?:/\d+\*?)+)(?:\((\d+)\))?')

    # Updated regex to correctly capture moves, including hits (*) and repetitions (n)
    move_pattern = re.compile(r'(bar|\d+)((?:/\d+\*?|/off)+)(?:\((\d+)\))?')

    moves = []
    
    for move in gnubg_moves.split():
        match = move_pattern.match(move)
        if match:
            start = match.group(1)  # Initial position (bar or number)
            move_sequence = match.group(2)  # All move steps including hits (*) and "off"
            count = int(match.group(3)) if match.group(3) else 1  # Extract count if (n) exists, otherwise 1
            
            start = 25 if start == "bar" else int(start)  # Convert "bar" to 25
            steps = re.findall(r'/(\d+|off)\*?', move_sequence)  # Extract all end positions
            
            for _ in range(count):  # Handle repeated moves
                temp_start = start
                for step in steps:
                    step = 0 if step == "off" else int(step)  # Convert "off" to 0
                    moves.append((temp_start, step))
                    temp_start = step  # Update start for next step
    
    return moves

# print(convert_move_notation("8/2*/off", [6,4]))
def correct_moves(parsed_moves):
    # print("Correcting:", parsed_moves)
    # print("Valid:", valid_moves)
    for i in range(len(parsed_moves)):
        start, end = parsed_moves[i]
        if start == 25:
            start=25
        else:
            start -=1
        if end == 0:
            end =27
        else:
            end -=1
        parsed_moves[i] = (start, end)
    return parsed_moves
        

def sanitise_move(move):
    return list_to_str([i for i in move if i!="*"], False, False)

if __name__ == "__main__":
    # random_cube_decisions()
    print("Activated")
    # print(gnubg_move([3,0,2,1,3,-1,0,2,-1,0,0,-2,-1,0,0,0,-2,0,-2,1,-2,-2,-2,2,0,1,0,0], [4,4], -1))
    # print(gnubg_move(make_board(), [3,1], 1))
    # print(gnubg_move(make_board(), [3,1], -1))
    # print(gnubg_offer_double(make_board(), None, 1))
    # print(gnubg_offer_double(make_board(), None, -1))
    # board = [4,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,-4,-2,-2,-3,-2,-2,0,0,0,8]
    # write_optimal_doubling(board)
    # write_optimal_doubling(invert_board(board))
    # print(get_optimal_doubling(board))
    # print(get_optimal_doubling(invert_board(board)))
    # random_cube_decisions()
    # print(write_optimal_doubling([-1, -10, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, -3, 4, 0, 1, 0, 0, 0, -1, 0, 0, 0], 0))
    # random_board_equities()
    # print(convert_move("16/15 16/14*",[2,1]))
    # print(transform_board([-1, -10, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, -3, 4, 0, 1, 0, 0, 0, -1, 0, 0, 0]))
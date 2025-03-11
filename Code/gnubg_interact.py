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
    print(board)
    gnuboard = [0]*26
    gnuboard[0] = board[25]
    gnuboard[25] = board[24]
    for i in range(1,25):
        gnuboard[i] = board[i-1] #24-i
    print(gnuboard)
    str_board = ""
    for i in gnuboard:
        str_board += f"{i} "
    print(str_board)
    return str_board.strip()

def get_move(board, roll):
    """Calls GNUBG and returns output of hint
    
    Args:
        board (list(int)): Board representation
        roll (list(int)): The dice roll

    Returns:
        str: The raw output of GNUBG
    """
    
    # Code from https://github.com/nitzankoll/backGammon-analyzer/blob/main/backgammon.py 
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
    
    try:
        print("Opening process...")
        # Start gnubg-cli process
        process = subprocess.Popen(
            ["gnubg","-t"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Communicate with gnubg-cli by sending commands
        print("Communicating...")
        stdout, stderr = process.communicate(commands, timeout=60)
        print("Terminating...")
        process.terminate()
        print("Waiting...")
        process.wait()
        # process.stdin.write("new game\n")
        # process.stdin.flush()
        # process.stdin.write("set player 1 name white\n")
        # process.stdin.flush()
        # process.stdin.write("set player 0 name black\n")
        # process.stdin.flush()
        # print("Setting board, turn and dice...")
        # process.stdin.flush()
        # process.stdin.write(f"set board simple {board}\n")
        # process.stdin.flush()
        # process.stdin.write("set turn 1\n")
        # process.stdin.flush()
        # process.stdin.write(f"set dice {roll[0]} {roll[1]}\n")
        # process.stdin.flush()
        # print("Calling hint...")
        # process.stdin.write("hint\n")
        # stdout = process.stdout.read()
        # print("Quitting...")
        # process.stdin.write("quit\n")
        # process.stdin.write("y\n")
        # process.stdin.close()
        # process.terminate()
        

        """
        # Debug output
        print("GNU Backgammon Output:")
        print(stdout)
        print("Errors (if any):")
        print(stderr)
        """
        return stdout
    except subprocess.TimeoutExpired:
        print("Process timed out, starting new one")
        process.kill()
        return None
    except FileNotFoundError:
        print("Error: gnubg-cli not found. Check the path.")
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

def convert_to_opengammon(board, raw_moves, roll):
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
        print("Move before conversion:", move_string)
        move = (convert_move(move_string, roll, board))
        moves.append(move)
        board_copy = board.copy()
        print(board_copy)
        pos_moves, _ = get_valid_moves(1, board, roll)
        if move not in pos_moves:
            print(f"{move} not possible")
            print(pos_moves)
            exit()
        for m in move:
            board_copy = update_board(board_copy, m)
        print(board_copy, move)
        boards.append(board_copy)
        if "(" in eq_str:
            eq_str, diff = eq_str.split("(")
        equities.append(float(eq_str.strip()))
    print("Converted to opengammon")
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
        print(move)
        if "off" in move:
            move = move.strip()
            if move[-1] == "f":
                move = move[:-3] + "28"
            else:
                index = move.index("o")
                move = move[:index] + "28"+ move[index+2:]
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
            for r in range(repeat):
                open_format.append(extract_base_move(base, roll, board))
        elif "*" in move:
            base = list_to_str([i for i in move if i != "*"], False, False)
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
            print(open_format)
            
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
    try:
        start, end = (int(start)-1, int(end)-1)
        # if start == 25: return (start, end)
    except ValueError:
        print(base_move)
        exit()
    if start - roll[0] > end and start - roll[1] > end:
        if not is_double(roll):
            ret = [()]*2
            if board[start - roll[0]] < -1:
                ret[0] = (start, start - roll[1])
                ret[1] = (start - roll[1], end)
            else:
                ret[0] = (start, start - roll[0])
                ret[1] = (start - roll[0], end)
            return ret
        else:
            diff = start - end
            moves = int(diff/roll[0])
            ret = [()]*moves
            for i in range(moves):
                ret[i] = (start - roll[0]*i, start-roll[0]*(i+1))
            return ret
    
    return start, end

def get_move_equities(board, roll, player):
    if player == -1:
        board = invert_board(board)
    transformed_board = transform_board(board)
    print("Getting moves...")
    gnu_output = get_move(transformed_board, roll)
    while gnu_output is None or is_double(roll):
        roll = roll_dice()
        gnu_output = get_move(transformed_board, roll)
    print("Parsing input...")
    raw_moves = hint_parser(gnu_output)
    print("Converting to opengammon...")
    return convert_to_opengammon(board, raw_moves, roll)

def write_move_equities(board, roll, player, i):
    moves, boards, equities = get_move_equities(board, roll, player)
    myFile = open(os.path.join("Data","Deep","GNUBG-data",f"positions{i-1}.txt"),"w")
    for i in range(len(moves)):
        myFile.write(f"{boards[i]},{equities[i]}\n")
    myFile.close()

def random_board_equities():
    positions = len(os.listdir(os.path.join("Data","Deep","GNUBG-data")))
    for i in range(100000):
        print(f"Number of boards generated: {i+positions}")
        board = generate_random_board()
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
            board = generate_random_board()
            moves, _ = get_valid_moves(1, board, roll)
        print(len(moves))
        write_move_equities(board, roll, 1, i+positions)
        

if __name__ == "__main__":
    # print(len(os.listdir(os.path.join("Data","Deep","GNUBG-data"))))
    # print("3/off"[:-4]+"/28")
    random_board_equities()
    # print(convert_move("16/15 16/14*",[2,1]))
    # print(transform_board([-1, -10, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, -3, 4, 0, 1, 0, 0, 0, -1, 0, 0, 0]))
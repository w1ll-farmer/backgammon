import subprocess
import os
import numpy as np
import re
from turn import *



def transform_board(board):
    gnuboard = [0]*26
    gnuboard[0] = board[25]
    gnuboard[25] = board[26]
    for i in range(1,25):
        gnuboard[i] = board[24-i]
    str_board = ""
    for i in gnuboard:
        str_board += f"{i} "
    return str_board.strip()
    
def get_move(board, roll):
    # Code from https://github.com/nitzankoll/backGammon-analyzer/blob/main/backgammon.py 
    commands = f"""
        new game
        set player 1 name white
        set player 0 name black
        set board simple {board}
        set turn {1}
        set dice {roll[0]} {roll[1]}
        hint
        quit
        y"""
    
    try:
        # Start gnubg-cli process
        process = subprocess.Popen(
            ["gnubg","-t"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Communicate with gnubg-cli by sending commands
        stdout, stderr = process.communicate(commands)

        """
        # Debug output
        print("GNU Backgammon Output:")
        print(stdout)
        print("Errors (if any):")
        print(stderr)
        """
        return stdout

    except FileNotFoundError:
        print("Error: gnubg-cli not found. Check the path.")
        return None

    return None

import re

def hint_parser(gnu_output):
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

def convert_to_opengammon(raw_moves):
    num_moves = len(raw_moves)
    moves, equities = [], []
    for i in range(num_moves):
        move_string, eq_str = raw_moves[i]
        

def convert_move(moves, roll):
    moves = moves.split()
    open_format = []
    for move in moves:
        if move.count('/') > 1: # One piece being moved only, hits at least one on the way
            base = list_to_str([i for i in move if i != "*"], False,False)
            base = base.split("/")
            for i in range(len(base)-1):
                base_move = extract_base_move(f"{base[i]}/{base[i+1]}",roll)
                if isinstance(base_move,list):
                    for i in range(len(base_move)):
                        open_format.append(base_move[i])
                else:
                    open_format.append(base_move)
        elif "(" in move:
            base, repeat = move.split("(")
            repeat = int(repeat.strip(')'))
            for r in range(repeat):
                open_format.append(extract_base_move(base, roll))
        elif "*" in move:
            base = list_to_str([i for i in move if i != "*"], False)
            base_move = extract_base_move(base, roll)
            if isinstance(base_move,list):
                for i in range(len(base_move)):
                    print(base_move[i])
                    open_format.append(base_move[i])
        else:
            base_move = extract_base_move(move, roll)
            if isinstance(base_move,list):
                for i in range(len(base_move)):
                    open_format.append(base_move[i])
            else:       
                open_format.append(base_move)
            
    return open_format

def extract_base_move(base_move, roll):
    start, end = base_move.split("/")
    start, end = (int(start), int(end))
    if start - roll[0] > end and start - roll[1] > end:
        if not is_double(roll):
            ret = [()]*2
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




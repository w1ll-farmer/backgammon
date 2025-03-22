import gym
from gym import spaces
import numpy as np
import main as backgammon
from turn import *
# from gnubg_interact import encode_board_vector

class BackgammonEnv(gym.Env):
    def __init__(self):
        super(BackgammonEnv, self).__init__()
        # self.action_space = spaces.Discrete()
        
    def get_reward(self, board, player):
        if not game_over(board):
            return 0
        reward = 3*is_backgammon(board) + 2*is_gammon(board)
        if reward == 0:
            reward = 1
        if abs(board[int(26.5+player/2)]) != 15:
            reward *= -1
        return reward
                       
        
class BackgammonGame:
    def __init__(self):
        self.board = make_board()
        self.current_player = 0
        self.done = False
        self.winner = None
        self.roll = roll_dice()
        
    def reset(self):
        self.board = make_board()
        self.current_player = 0
        self.done = False
        self.winner = None
        self.roll = roll_dice()
        return self.get_observation()
    
    def get_observation(self):
        vector = []
        for point in range(len(self.board)):
            vector += self.encode_point(point)
        vector.append(-self.board[24]/2)
        vector.append(self.board[25]/2)
        vector.append(-self.board[26]/15)
        vector.append(self.board[27]/15)
        vector.append(int(self.current_player == 1))
        vector.append(int(self.current_player == -1))
        return vector
        
    
    def encode_point(self, point):
        base = [0]* 8
        if point < 0:
            for i in range(0,3):
                if point < -i:
                    base[i] = 1
            if point < -3:
                base[3] = (abs(point)-3)/2 
        elif point > 0:
            for i in range(4, 7):
                if point > i -4:
                    base[i] = 1
            if point > 3:
                base[7] = (point-3)/2
        return base
        
    def get_legal_action(self):
        return get_valid_moves(self.current_player, self.board, self.roll)

game = BackgammonGame()
print(game.encode_point(make_board()[5]))
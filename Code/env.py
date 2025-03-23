import gym
from gym import spaces
import numpy as np
import reinforce_agent # OpenLearn
import main
import random_agent
from turn import *
import torch
from random import randint, uniform
# from gnubg_interact import encode_board_vector

class BackgammonEnv(gym.Env):
    def __init__(self, oppstrat):
        super(BackgammonEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(198,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(256)
        
        self.opponent = oppstrat
        self.current_player = 1
        self.current_board = make_board()
        self.valid_raw_boards = []
        self.valid_encoded_boards = []
        self.valid_moves = []
        self.roll = []
        
        self.done = False
        self.reward = 0
        self.w_score = 0
        self.b_score = 0
        
        self.time_step = 0
        
    def reset(self):
        self.current_board = make_board()
        self.roll = roll_dice()
        
        while self.roll[0] == self.roll[1]:
            self.roll = roll_dice()
        if self.roll[0] > self.roll[1]:
            self.current_player = -1
        else:
            self.current_player = 1
            
        self.done = False
        self.reward = 0
        self.w_score = 0
        self.b_score = 0
        self.time_step = 0
        return self._get_obs()
    
    def step(self, action_idx):
        if self.done:
            return self._get_obs(), 0, True, {}
        
        # Capture current state
        current_state = self._get_obs()
        
        if action_idx is not None:
            self.current_board = self.valid_raw_boards[action_idx]
            # print(self.current_board, "AGENT", self.roll, self.valid_moves[action_idx], self.current_player)
            current_state = self.valid_encoded_boards[action_idx]
        self.current_player *= -1
            
        # Get reward and check termination
        reward_vector, self.done = self._check_game_over()
        reward = self._get_reward_scalar(reward_vector)
        if reward_vector != [0]*6:
            print(f"{reward_vector}\n{reward}")
            print(self.current_board)
            
        
        # Get next state via opponent's turn
        next_state = None
        if not self.done:
            next_state = self.opponent_turn()  # Returns encoded state
        # Update weights using pre-encoded next_state
        model.update_weights(
            current_state=current_state,
            next_state=next_state,  # Directly use returned encoded state
            reward=reward,
            done=self.done
        )
    
        return next_state, reward, self.done, {"reward": reward_vector}
          
    def opponent_turn(self):
        self._start_turn()
        if len(self.valid_moves) > 0:
            if self.opponent == "RANDOM":
                # self.current_board, _ = random_agent.randobot_play(
                #     self.roll, self.valid_moves, self.valid_raw_boards
                # )
                idx = randint(0, len(self.valid_raw_boards)-1)
                self.current_board = self.valid_raw_boards[idx]
            # print(self.current_board, "Random", self.roll, self.valid_moves[idx], self.current_player)
        self.update_time_step()
        self.current_player *= -1
        return self._encode_state(self.current_board)
        
        
    def _calculate_score(self, player):
        if self.current_board[int(26.5+player*0.5)] != 15:
            return 0
        if is_backgammon(self.current_board):
            return 3
        elif is_gammon(self.current_board):
            return 2
        else:
            return 1
    
    def _get_reward_scalar(self, reward_vector):
        return sum(reward_vector) if reward_vector[0] == 1 else -sum(reward_vector)
    
    
    def _encode_state(self, board_state):
        vector = []
        for point in range(24):
            vector += self._encode_point(board_state[point])
        vector.append(abs(board_state[24]/2))
        vector.append(board_state[25]/2)
        vector.append(abs(board_state[26]/15))
        vector.append(board_state[27]/15)
        vector.append(int(self.current_player == 1))
        vector.append(int(self.current_player == -1))
        return torch.FloatTensor(vector)
    
    def _start_turn(self):
        self.roll = roll_dice()
        self._get_valid_moves()
    
    def _get_valid_moves(self):
        self.valid_moves, self.valid_raw_boards =  get_valid_moves(
            self.current_player, self.current_board, self.roll
        )
        if len(self.valid_raw_boards) > 0:
            self.valid_encoded_boards = [
                self._encode_state(board) for board in self.valid_raw_boards
            ]
       
    def _check_game_over(self):
        """Determine reward vector and termination"""
        if not game_over(self.current_board):
            return [0, 0, 0, 0, 0, 0], False
        # print("current player",self.current_player)
        if self.current_board[27] == 15:
            # print("Player 1 win")
            if is_backgammon(self.current_board):
                return [1, 0, 1, 0, 1, 0], True
            if is_gammon(self.current_board):
                return [1, 0, 1, 0, 0, 0], True
            return [1, 0, 0, 0, 0, 0], True
        else:
            if is_backgammon(self.current_board):
                return [0, 1, 0, 1, 0, 1], True
            if is_gammon(self.current_board):
                return [0, 1, 0, 1, 0, 0], True
            return [0, 1, 0, 0, 0, 0], True
    
    def _encode_point(self, point):
        base = [0]* 8
        if point < 0:
            for i in range(0,3):
                if point < -i:
                    base[i] = 1
            if point < -3:
                base[3] = min((abs(point)-3)/2,1)
        elif point > 0:
            for i in range(4, 7):
                if point > i -4:
                    base[i] = 1
            if point > 3:
                base[7] = min((point-3)/2,1)
        return base
    
    def _get_obs(self):
        return self._encode_state(self.current_board)
    
    def update_time_step(self):
        self.time_step += 0.5


def save_model(model, episode, path="backgammon_model.pth"):
    """Save model checkpoint"""
    torch.save({
        'episode': episode,  # Current training episode
        'model_state_dict': model.state_dict(),  # Model parameters
        'eligibility_traces': model.eligibility_traces  # Eligibility traces
    }, path)
    print(f"Model saved to {path}")
    
model = reinforce_agent.ReinforceNet()
env = BackgammonEnv("RANDOM")

# agent_score = 0
# opp_score = 0
for episode in range(1,201):
    # agent_score += env.w_score
    # opp_score += env.b_score
    # print(f"\n {agent_score}-{opp_score}")
    state = env.reset()
    done = False
    if env.current_player == -1:
        env._get_valid_moves()
        env.opponent_turn()
    
    while not done:
        if env.time_step == 0:
            env._get_valid_moves()
        else:
            env._start_turn()
        
        if len(env.valid_raw_boards) > 0:
            # Inside training loop
            board_tensors = torch.FloatTensor(np.array(env.valid_encoded_boards))
            # outcome_probs = []
            with torch.no_grad():
                outcome_probs = model(board_tensors)  # Shape: [num_boards, 6]
                # for board_tensor in board_tensors:
                    # outcome_probs += model(board_tensor)
                expected_values = model.expected_value(outcome_probs)  # Shape: [num_boards]

            action_idx = torch.argmax(expected_values).item()  # Pick board with highest expected value
            # action_idx = randint(0, len(env.valid_raw_boards)-1)
            # env.current_board = env.valid_raw_boards[action_idx]
        else:
            action_idx = None            
        next_state, reward, done, _ = env.step(action_idx)
        
    if episode % 100 == 0:
        save_model(model, episode, path=f"Code/RL/reinforcement_{episode}.pth")
        
        
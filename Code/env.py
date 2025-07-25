import gym
from gym import spaces
import numpy as np
import reinforce_agent # OpenLearn
from turn import *
from greedy_agent import easy_evaluate, evaluate
from expectimax_agent import expectimax_play
import torch
from random import randint, uniform
from testfile import invert_board
import copy
import main
# from gnubg_interact import encode_board_vector

class BackgammonEnv(gym.Env):
    def __init__(self, oppstrat):
        super(BackgammonEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(289,), dtype=np.float32
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
        """Starts a new game

        Returns:
            Tensor: Encoded board state
        """
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
        """Plays through one player's turn

        Args:
            action_idx (int): The index of the chosen move

        Returns:
            (next_state,reward,done,reward): Variables needed for updating weights
        """
        if self.done:
            return self._get_obs(), 0, True, {}

        # Capture current state BEFORE making a move
        current_state = self._get_obs()  

        # Apply the move if action_idx is valid
        if action_idx is not None:
            self.current_board = self.valid_raw_boards[action_idx]  # Apply the move
            # move = self.valid_moves[action_idx]
            # VALIDATION PRINTOUT
            # print(f"Player {self.current_player} {self.current_board} {move} {self.roll}")
        # Switch player
        self.current_player *= -1

        # Check for game over and compute reward
        reward_vector, self.done = self._check_game_over()
        reward = self._get_reward_scalar(reward_vector)

        # Capture next state AFTER the move is applied
        next_state = self._get_obs() if not self.done else None

        # Update model weights
        model.update_weights(
            current_state=current_state,  # BEFORE the move
            next_state=next_state,  # AFTER the move
            reward=reward,
            done=self.done
        )

        return next_state, reward, self.done, {"reward": reward_vector}

        
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
    
    
    def _encode_state(self, board_state, player=None):
        """Encodes the board state into input vector for model

        Args:
            board_state (list(int)): Raw board state
            player (int, optional): Player who's perspective board is seen thru. Defaults to None.

        Returns:
            _type_: _description_
        """
        if player is None:
            player = self.current_player
        elif player == 1 and self.current_player == -1:
            board_state = invert_board(board_state)
        return torch.Tensor(convert_board(board_state, False, cube=True, RL=True, player=player))
    #     vector = []
    #     for point in range(24):
    #         vector += self._encode_point(board_state[point])
    #     vector.append(abs(board_state[24]/2))
    #     vector.append(board_state[25]/2)
    #     vector.append(abs(board_state[26]/15))
    #     vector.append(board_state[27]/15)
    #     vector.append(int(player == 1))
    #     vector.append(int(player == -1))
    #     return torch.FloatTensor(vector)
    
    def _start_turn(self):
        self.roll = roll_dice()
        if self.current_player == -1:
            self._get_valid_moves(1)
        else:
            self._get_valid_moves()
    
    def _get_valid_moves(self, inverted_player=None):
        if inverted_player is None:
            inverted_player = self.current_player
        self.valid_moves, self.valid_raw_boards =  get_valid_moves(
            self.current_player, self.current_board, self.roll
        )
        if len(self.valid_raw_boards) > 0:
            self.valid_encoded_boards = [
                self._encode_state(board, inverted_player) for board in self.valid_raw_boards
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
            elif is_gammon(self.current_board):
                return [1, 0, 1, 0, 0, 0], True
            return [1, 0, 0, 0, 0, 0], True
        else:
            if is_backgammon(self.current_board):
                return [0, 1, 0, 1, 0, 1], True
            elif is_gammon(self.current_board):
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

def load_model(model, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eligibility_traces = checkpoint['eligibility_traces']
    print(f"Model loaded from {path} (episode {checkpoint['episode']})")
    
def benchmark(episode):
    agent_score, opp_score = 0,0
    weights1 = f"V3_{episode}"
    for i in range(200):
        # w_vector, w_score, b_vector, b_score = main.backgammon(1, "REINFORCEMENT", weights1, "REINFORCEMENT", "self_111000", cube_on=False)
        w_vector, w_score, b_vector, b_score = main.backgammon(1, "REINFORCEMENT", weights1, "GREEDY", None, cube_on=False, w_lookahead=False, b_lookahead=False)
        agent_score += w_score
        opp_score += b_score
        if i % 50 == 0: print(f"{i} games: {agent_score}-{opp_score}")
    print(f"Episode {episode}: {agent_score}-{opp_score} v Greedy")
    diff = str(agent_score - opp_score)
    ep = str(episode)
    myFile = open(os.path.join("Data","RL","benchmark3.txt"),"a")
    myFile.write(f"{ep},{diff}\n")
    myFile.close()

def bench_genetic(episode):
    agent_score, opp_score = 0,0
    weights1 = f"V3_{episode}"
    for i in range(200):
        # w_vector, w_score, b_vector, b_score = main.backgammon(1, "REINFORCEMENT", weights1, "REINFORCEMENT", "self_111000", cube_on=False)
        w_vector, w_score, b_vector, b_score = main.backgammon(1, "REINFORCEMENT", weights1, "GENETIC", [10.0, 21.0, 12.0, 11.0, 15.0, 0.5664383320165035, 10.0, 4.0, 25.0, 6.0, 0.6461166029382669, 0.5378085318259279, 0.5831066576570856, 0.9552318750278183, 0.07412843879077036, 0.17550708535892934, 0.49191128795644823, 0.556755495835094], cube_on=False, w_lookahead=False, b_lookahead=False)
        agent_score += w_score
        opp_score += b_score
        if i % 50 == 0: print(f"{i} games: {agent_score}-{opp_score}")
    print(f"Episode {episode}: {agent_score}-{opp_score} v Genetic")
    diff = str(agent_score - opp_score)
    ep = str(episode)
    myFile = open(os.path.join("Data","RL","benchmark3genetic.txt"),"a")
    myFile.write(f"{ep},{diff}\n")
    myFile.close()

def bench_V2(episode):
    agent_score, opp_score = 0,0
    weights1 = f"V3_{episode}"
    for i in range(200):
        # w_vector, w_score, b_vector, b_score = main.backgammon(1, "REINFORCEMENT", weights1, "REINFORCEMENT", "self_111000", cube_on=False)
        w_vector, w_score, b_vector, b_score = main.backgammon(1, "REINFORCEMENT", weights1, "REINFORCEMENT", "self_170000", cube_on=False, w_lookahead=False, b_lookahead=False)
        agent_score += w_score
        opp_score += b_score
        if i % 50 == 0: print(f"{i} games: {agent_score}-{opp_score}")
    print(f"Episode {episode}: {agent_score}-{opp_score} v RLv2")
    diff = str(agent_score - opp_score)
    ep = str(episode)
    myFile = open(os.path.join("Data","RL","benchmark3RL2.txt"),"a")
    myFile.write(f"{ep},{diff}\n")
    myFile.close()

def bench_gnubg(episode):
    agent_score, opp_score = 0,0
    weights1 = f"V3_{episode}"
    for i in range(200):
        # w_vector, w_score, b_vector, b_score = main.backgammon(1, "REINFORCEMENT", weights1, "REINFORCEMENT", "self_111000", cube_on=False)
        w_vector, w_score, b_vector, b_score = main.backgammon(1, "GNUBG", None, "REINFORCEMENT", weights1, cube_on=False, w_lookahead=False, b_lookahead=False)
        agent_score += b_score
        opp_score += w_score
        if i % 50 ==0: print(f"{i} games: {agent_score}-{opp_score}")
    print(f"Episode {episode}: {agent_score}-{opp_score} v Intermediate")
    diff = str(agent_score - opp_score)
    ep = str(episode)
    myFile = open(os.path.join("Data","RL","benchmark3intermediate.txt"),"a")
    myFile.write(f"{ep},{diff}\n")
    myFile.close()
    
model = reinforce_agent.ReinforceNet3()
start_point = 32000

load_model(model, os.path.join("Code","RL",f"reinforcement_V3_{start_point}.pth"))
model.epsilon = model.epsilon**(1+start_point // 50000)
print(f"Epsilon = {model.epsilon}")
env = BackgammonEnv("RANDOM")
# bench_genetic(start_point)
for episode in range(1+start_point, 1000001):
    state = env.reset()
    model.reset_eligbility_traces()
    done = False
    time_steps = 0
    while not done and time_steps < 2500:
        if time_steps > 0:
            env._start_turn()  # Get valid moves, roll
        else:
            env._get_valid_moves(1) # Get valid moves

        if len(env.valid_raw_boards) > 0:
            # Both players use the same model to decide moves
            # print(f"Len valid encoded boards: {len(env.valid_encoded_boards)}")
            action_idx = model.select_action(env.valid_encoded_boards)
        else:
            action_idx = None  # No valid move, must pass

        # Step forward in the game and update model *after every move*
        next_state, reward, done, _ = env.step(action_idx)

        time_steps += 1
    if time_steps >= 500: print(f"Reached {time_steps} time steps")
    if episode % 50 == 0:
        print(f"Episode {episode}")
    if episode % 500 == 0:
        save_model(model, episode, path=f"Code/RL/reinforcement_V3_{episode}.pth")
    if episode % 1000 == 0:    
        bench_genetic(episode)
    if episode % 5000 == 0:
        bench_V2(episode)
    if episode % 10000 == 0:
        bench_gnubg(episode)
    if episode % 50000 == 0:
        model.epsilon *= 0.1
        print(f"Epsilon = {model.epsilon}")
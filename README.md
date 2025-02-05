# Backgammon 

## Board description
 - Len(board) = 28
 - If $board[i]$ is unoccupied, $board[i]=0$
 - If $board[i]$ is occupied, $board[i] = player*number of pieces occupying point$
    - For example, if player 1 has 4 pieces on item 20, $board[20] = 4$
 - Player 1's home contains items 0-5. If hit, goes to 25. If borne off, goes to 27
 - Player -1's contains items 18-23. If hit, goes to 24. If borne off, goes to 26
 
## AIs
 - whitestrat (player 1's strategy) and blackstrat (player -1's strategy)
 - At the start of each game the player with the higher initial roll becomes player1 (they start first)
 - At the start of each new round, the winner of the previous round starts first and becomes player1
 - This is done by using player1strat and player2strat


## Global Variables and Constants
 - commentary: outputs the board state, what moves are taken, whose turn it is
 - GUI_FLAG: whether or not to display the GUI. Large chunks of code are activated by this flag
 - test: whether or not to run tests during runtime
 - background: the background of the GUI
 - w_score: The score of the white player (1). Updated at the end of each round
 - b_score: The score of the black player (-1). Updated at the end of each round


### Turn
- Contains all backend functions
    - roll_dice
    - make_board
    - print_board
    - is_double
    - update_board
    - all_past
    - get_home_info
    - must_enter
    - can_enter
    - all_checkers_home
    - get_legal_moves
    - get_valid_moves
    - game_over
    - is_backgammon
    - is_gammon
    - is_error

### genetic_train
- Trains a genetic agent. Not necessary for debugging
    - get_parent
    - reproduce
    - calc_fitness
    - calc_overall_fitness
    - mutate
    - generate_initial_pop
    - genetic
    - co_evolve

### greedy_agent
- Evaluates moves for the greedy agent to choose
    - count_blots
    - count_walls
    - eevaluate (easy evaluate, not for use on dissertation)
    - evaluate (actual evaluation function, in use for dissertation)

### gui
- Not necessary for debugging

### random_agent
- AI that makes random moves
    - generate_random_move
    - randobot_play

### testfile
 - Contains all isolated tests
    - check_inverted
    - invert_board
    - check_moves

from greedy_agent import evaluate
from turn import get_valid_moves

def expectimax_play(moves, boards, player):
    # Enumerate each board in boards
    # Check each resulting board via evaluate function and find max value
    # Choose board state that minimises the resulting max value
    best_board = []
    best_score = float("inf")
    for board in boards:
        board_score = 0
        for die1 in range(1,7):
            for die2 in range(1, 7):
                max_die_score = 0
                roll = sorted([die1, die2])
                adv_moves, adv_boards = get_valid_moves(-player, board, roll)
                for i in range(len(adv_boards)):
                    s = evaluate(board, adv_boards[i], -player)
                    if s > max_die_score: max_die_score = s
                board_score += max_die_score
        if board_score < best_score:
            best_score = board_score
            best_board = board
    chosen_move = moves[boards.index(best_board)]
    return [chosen_move], best_board
def adaptive_play(moves, boards, player, roll, board):
    pass

def calc_equity(pip_diff, home_board_points, blots, prime_length, gammons):
    # Assign heuristic weights (tuned via testing)
    pip_weight = -0.02  # Losing pip count is bad
    home_board_weight = 0.05
    blot_penalty = -0.1  # Each blot is risky
    prime_weight = 0.08  # Strong primes are valuable
    gammon_weight = 0.15  # Gammon chances matter

    # Compute heuristic equity estimate
    equity = (
        pip_weight * pip_diff +
        home_board_weight * home_board_points +
        blot_penalty * blots +
        prime_weight * prime_length +
        gammon_weight * gammons
    )

    # Clamp equity to [-1, 1] range
    return max(-1, min(1, equity))

# Example Usage: Player is ahead in race, strong home board, but has 2 blots
equity = calc_equity(pip_diff=-10, home_board_points=4, blots=2, prime_length=3, gammons=0.2)
print(f"Estimated Equity: {equity:.3f}")

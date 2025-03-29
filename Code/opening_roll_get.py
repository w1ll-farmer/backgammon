from turn import *
from main import deep_play, greedy_play
def deep_opening_roll():
    starting_board = make_board()
    player = 1
    MyFile = open(f"Data/OpeningRolls/geneticopeningrollselection.txt","a")
    print(os.listdir())
    for roll1 in range(1, 7):
        for roll2 in range(roll1, 7):
            roll = [roll1, roll2]
            moves, boards = get_valid_moves(1, starting_board, roll)
            move, board, _ = greedy_play(moves, boards, starting_board, 1, roll, [10.0, 21.0, 12.0, 11.0, 15.0, 0.5664383320165035, 10.0, 4.0, 25.0, 6.0, 0.6461166029382669, 0.5378085318259279, 0.5831066576570856, 0.9552318750278183, 0.07412843879077036, 0.17550708535892934, 0.49191128795644823, 0.556755495835094])
            MyFile.write(f"{roll1},{roll2},{move.pop()}\n")
    MyFile.close()

deep_opening_roll()
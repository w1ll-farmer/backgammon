from turn import *
from main import deep_play, greedy_play
from double import *
from reinforce_play import reinforce_play
from adaptive_agent import race_gwc
global adaptive_weights
adaptive_weights = [0.9966066885314592, -0.9916984096898946, 0.3106830724424913, 0.529168163359478, -0.4710732676896102, 0.5969523488654117, 0.36822981983332415, 0.38958074063216697, 0.02676397245530815, 0.08588282381449319, 0.06094873757931751, 1.1095422351658368, 0.47764793610307643, 0.040753486445243126, 0.5495226441839489, 0.8875009606764003, 0.9333344067224983, 0.1340269726805713, 0.1978868967026618, 1.2096547126804458, 2.379707426788366, 0.6465298771549699, 0.509196585225148, 0.261875669397977, 0.36883752029556166, -0.481342015629518, 0.7098436807557322, 1.0250219115287624, 0.5739284594183071, 0.1796876959733017, 0.2679991261065485]
def deep_opening_roll():
    starting_board = make_board()
    player = 1
    MyFile = open(f"Data/OpeningRolls/reinforcementopeningrollselection.txt","a")
    print(os.listdir())
    for roll1 in range(1, 7):
        for roll2 in range(roll1, 7):
            roll = [roll1, roll2]
            moves, boards = get_valid_moves(1, starting_board, roll)
            move, board= reinforce_play(boards, moves, 1, "self_170000")
            MyFile.write(f"{roll1},{roll2},{move}\n")
    MyFile.close()

def get_double_data(race=False):
    cubePath = os.path.join("Data","Deep","GNUBG-data","Cube")
    if race:
        cubePath = os.path.join(cubePath, "Race")
    acceptFile = open(os.path.join(cubePath,"Accept","positions.txt"),"r")
    acceptset = []
    for line in acceptFile:
        board, decision = line.split("],")
        decision = int(decision)
        board = board[1:]
        board = [int(i) for i in board.split(",")]
        acceptset.append((board, decision))
    acceptFile.close()
    offerFile = open(os.path.join(cubePath,"Offer","positions.txt"),"r")
    offerset = []
    for line in offerFile:
        board, decision = line.split("],")
        decision = int(decision)
        board = board[1:]
        board = [int(i) for i in board.split(",")]
        offerset.append((board, decision))
    offerFile.close()
    return acceptset, offerset
# Accept accuracy: 58.77478233504668% double point: 1.7747723546862937
def test_double(acceptset, offerset, strat, race, test = False):
    accept_point = uniform(1, 1.5) if test else 0.47946108471546767 if strat == "ADAPTIVE" else 0.8
    correct, total = 0,0
    for i in range(len(acceptset)):
        board, decision = acceptset[i]
        # print(decision)
        if strat == "ADAPTIVE":
            if race:
                gwc = race_gwc(board, 1)
                output = gwc >= 0.2
            else:
                equity = calc_advanced_equity(board, 1, 0, 0, 1, 25, adaptive_weights)
                output = advanced_accept_double(equity, accept_point)
        elif strat == "REINFORCEMENT":
            output = reinforce_accept_double(board, -1, "self_170000", accept_point)
            # if test:
                # with open(os.path.join("Data","RL","Doubling","foldpoint.txt"),'a') as myFile:
                #     myFile.write(f"{output},{decision}\n")
        else:
            equity = calc_equity(board, 1)
            output = basic_accept_double(equity)
        total += 1
        correct += int(int(output) == decision)
    print(f"Accept accuracy: {100*correct/total}% double point: {accept_point}")
    if not test:
        correct, total = 0,0
        for i in range(len(offerset)):
            board, decision = offerset[i]
            # print(decision)
            if strat == "ADAPTIVE":
                if race:
                    gwc = race_gwc(board, 1)
                    output = gwc >= 0.8
                else:
                    equity = calc_advanced_equity(board, 1, 0, 0, 1, 25, adaptive_weights)
                    output = advanced_should_double(equity)
            elif strat == "REINFORCEMENT":
                output = reinforce_should_double(board, 1, "self_170000")
            else:
                equity = calc_equity(board, 1)
                output = basic_should_double(equity)
            total += 1
            correct += int(int(output) == decision)
            
        print(f"Offer accuracy: {100*correct/total}%")

acceptset, offerset = get_double_data(True)
test_double(acceptset, offerset, "REINFORCEMENT", True, False)

# for i in range(100):
#     test_double(acceptset, offerset, "REINFORCEMENT", False, True)
# print("ONTO RACE")
# for i in range(100):
#     test_double(acceptset, offerset, "REINFORCEMENT", True, True)
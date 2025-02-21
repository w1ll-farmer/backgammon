from random import randint, uniform, gauss
from main import *
from time import sleep

def write(individual, strat):
    strat = strat.lower()
    file = open(f"./Data/{strat}-fit.txt","a")
    file.write(f"{individual}\n")
    file.close()
    
    
def get_parent(P, population_fitness):
    """Using the wheel to find parent for breeding

    Args:
        P (list(tuple(float))): The population of different weights and scores
        population_fitness (list(float)): The fitness of each i in population
    Returns:
        tuple(float): The selected parent
    """
    total_fitness = sum(population_fitness)
    threshold = uniform(0,1)*total_fitness
    cumulative_fitness = 0
    i = 0
    while cumulative_fitness < threshold:
        cumulative_fitness += population_fitness[i]
        i += 1
    if i >= len(P):
        return P[-1]
    return P[i]

def reproduce(mother, father, strat):
    crossover_point = randint(0,len(mother)-1)
    child1 = mother[:crossover_point] + father[crossover_point:]
    child2 = father[:crossover_point] + mother[crossover_point:]
    children = [child1, child2]
    children_fitness = [calc_fitness(child, strat) for child in children]
    return children[children_fitness.index(max(children_fitness))], max(children_fitness)
    # return max(children, key=lambda x: calc_fitness(x))


def calc_fitness(individual, strat, first_to=13):
    oppstrat = "GREEDY" if strat == "GENETIC" else "GENETIC"
    opp_weights = None if strat == "GENETIC" else [10.0, 21.0, 12.0, 11.0, 15.0, 0.5664383320165035, 10.0, 4.0, 25.0, 6.0, 0.6461166029382669, 0.5378085318259279, 0.5831066576570856, 0.9552318750278183, 0.07412843879077036, 0.17550708535892934, 0.49191128795644823, 0.556755495835094]
    _, g_score, _, opp_score = backgammon(first_to, strat ,individual, oppstrat, opp_weights)
    try:
        fitness = g_score / opp_score
    except ZeroDivisionError:
        fitness = g_score + 1
    return fitness


def calc_overall_fitness(P, strat):
    population_fitness = []
    for individual in P:
        population_fitness.append(calc_fitness(individual, strat, 5))
    return population_fitness


def mutate(child):
    # randomly adjust points allocation from greedy algorithm? 
    swap_1_index = randint(0, len(child)-1)
    temp = child[swap_1_index]
    swap_2_index = randint(0, len(child)-1)
    
    child[swap_1_index] = child[swap_2_index]
    child[swap_2_index] = temp
    for i in range(len(child)):
        child[i] = gauss(child[i], 0.288)
    return child 


def generate_initial_pop(pop_size, strat):
    """Get weights for different moves types for whole population

    Args:
        pop_size (int): The number of individuals in the population

    Returns:
        list(tuple(float)): Individuals in population
    """
    P = []
    for i in range(pop_size):
        # Evaluation scores for each contingency
        if strat == "GENETIC":
            walled_off = randint(0,27)
            walled_off_hit = randint(0,27)
            borne_off_add = randint(0,27)
            bear_off_points = randint(0,27)
            hit_off_points = randint(0,27)
            hit_off_mult = uniform(0,1)
            exposed_hit = randint(0,27)
            wall_blot_home_points = randint(0,27)
            wall_points = randint(0,27)
            blot_points = randint(0,27)
            home_points = randint(0,27)
            wall_mult = uniform(0,1)
            blot_mult = uniform(0,1)
            home_mult = uniform(0,1)
            blot_diff_mult = uniform(0,1)
            wall_diff_mult = uniform(0,1)
            wall_maintain = uniform(0,1)
            blot_maintain = uniform(0,1)
            P.append([walled_off, walled_off_hit, borne_off_add,
                bear_off_points, hit_off_points, hit_off_mult,
                exposed_hit, wall_blot_home_points, wall_points,
                blot_points, home_points, wall_mult, blot_mult,
                home_mult, blot_diff_mult, wall_diff_mult,
                wall_maintain, blot_maintain])
        else:
            pip_weight = uniform(0,1)  # Increase importance of race in later game
            blot_penalty = -uniform(0,1)  # Harsher penalty for blots that can be hit
            prime_weight = uniform(0,1)  # Increased impact of prime structures
            home_board_weight = uniform(0,1)  # Stronger boards are more valuable
            opp_home_board_penalty = -uniform(0,1)  # Opponent's home board makes hits worse
            gammon_weight = uniform(0,1)  # Increased weight for gammon potential
            bearing_off_weight = uniform(0,1)  # Make bearing off more valuable
            cube_volatility_weight = uniform(0,1)
            pip_diff_normaliser = uniform(0,1)
            prime_bonus = uniform(0,1)
            bear_off_normaliser = uniform(0,1)
            win_volatility = uniform(1,3)
            catch_up_volatility=uniform(0,1)
            blot_volatility=uniform(0,1)
            hit_volatility=uniform(0,1)
            opp_home_volatility=uniform(0,1)
            opp_piece_home_potential=uniform(0,1)
            opp_piece_player_potential=uniform(0,1)
            bear_off_potential=uniform(0,1)
            win_bonus=uniform(1,3)
            catch_up_bonus=uniform(1,3)
            close_bonus=uniform(0,2)
            no_chance_bonus=uniform(0,1)
            default=uniform(0,1)
            wall_mult=uniform(0,1)
            home_mult=-uniform(0,1)
            reentry_mult=uniform(0,1)
            exposed_mult=uniform(0,1)
            safe_mult=uniform(0,1)
            vulnerable_mult=uniform(0,1)
            volatility_normaliser=uniform(0,1)
            P.append([pip_weight, blot_penalty, prime_weight, home_board_weight, opp_home_board_penalty,
                gammon_weight, bearing_off_weight, cube_volatility_weight, pip_diff_normaliser,
                prime_bonus, bear_off_normaliser,  win_volatility, catch_up_volatility,
                blot_volatility, hit_volatility, opp_home_volatility, opp_piece_home_potential,
                opp_piece_player_potential, bear_off_potential, win_bonus, catch_up_bonus,
                close_bonus, no_chance_bonus, default, wall_mult, home_mult, reentry_mult,
                exposed_mult, safe_mult, vulnerable_mult, volatility_normaliser])
        
    return P
    
    
def genetic(max_iters, pop_size, strat):
    """Runs the genetic algorithm

    Args:
        max_iters (int): Max number of generations
        pop_size (int): Population size
    
    Returns:
        list(int): Fittest individual's weights
    """
    P = generate_initial_pop(pop_size, strat)
    print("Generated initial population")
    fittest = P[0]
    fittest_fitness = calc_fitness(fittest, strat)
    print("Calculating overall fitness")
    population_fitness = calc_overall_fitness(P, strat)
    # iterate through max_it times
    for iteration in range(max_iters):
        print(f"Iterations: {iteration}/{max_iters}")
        newP = []
        for i in range(pop_size):
            X = get_parent(P, population_fitness)
            Y = get_parent(P, population_fitness)
            Z, Z_fitness = reproduce(X, Y, strat)
            if randint(1, 100) > 95:
                Z = mutate(Z)
                Z_fitness = calc_fitness(Z, strat)
            if Z_fitness >= 2:
                write(str(Z), strat)
            if Z_fitness > fittest_fitness:
                fittest_fitness = Z_fitness
                fittest = Z
                print(fittest, fittest_fitness)
            newP.append(Z)
        P = newP
        print('Fittest',fittest, fittest_fitness)
    fittest = co_evolve(strat)
    return fittest

def co_evolve(strat):
    file = open(f"./Data/{strat.lower()}-fit.txt","r")
    P = []
    for line in file:
        individual = line.strip()
        individual = individual.strip("[")
        individual = individual.strip("]")
        individual = [float(i) for i in individual.split(",")]
        P.append(individual)
    j =0
    while len(P) > 1:
        newP = []
        i = 0
        # first_to = [5, 9, 13, 25] + [25]*len(P)
        first_to = [25]*(1+len(P))
        while i < len(P) - 1:
            print(f"Round {i//2} out of {len(P)//2}")
            p1wins, p2wins = 0,0
            while p1wins < 1 and p2wins < 1:
                print(f"First to 25, {strat}")
                _, p1score, _, p2score = backgammon(25, strat, P[i], strat,P[i+1])
                if p1score > p2score:
                    p1wins += 1
                else: 
                    p2wins += 1
            if p1wins > p2wins:
                print("winner",P[i])
                newP.append(P[i])
            else:
                newP.append(P[i+1])
                print("winner",P[i+1])
            if i+3 >= len(P) and i+2 < len(P):
                newP.append(P[i+2])
            i += 2
        P = newP
        j +=1
    return P[0]


print(genetic(10, 50, "ADAPTIVE"))
# print(co_evolve("ADAPTIVE"))
# [0.6219952084521901, 27.0, 4.0, 26.0, 0.46015349243263104, 0.713687637052133, 7.0, 2.0, 26.0, 4.0, 0.0, 0.6337036278226582, 0.15012449622656665, 0.5226624630505539, 0.7313044431665402, 0.6662731224336713, 0.667683543270852, 0.906174549240715]

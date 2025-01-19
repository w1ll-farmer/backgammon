from random import randint, uniform
from main import *
from time import sleep

def write(individual):
    file = open("./Data/genetic-fit","a")
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

def reproduce(mother, father):
    crossover_point = randint(0,len(mother)-1)
    child1 = mother[:crossover_point] + father[crossover_point:]
    child2 = father[:crossover_point] + mother[crossover_point:]
    children = [child1, child2]
    return max(children, key=lambda x: calc_fitness(x))


def calc_fitness(individual):
    _, g_score, _, opp_score = backgammon(5, "GENETIC",individual, "GREEDY")
    try:
        fitness = g_score / opp_score
    except ZeroDivisionError:
        fitness = g_score + 1
    return fitness


def calc_overall_fitness(P):
    population_fitness = []
    for individual in P:
        population_fitness.append(calc_fitness(individual))
    return population_fitness


def mutate(child):
    # randomly adjust points allocation from greedy algorithm? 
    swap_1_index = randint(0, len(child)-1)
    temp = child[swap_1_index]
    swap_2_index = randint(0, len(child)-1)
    
    child[swap_1_index] = child[swap_2_index]
    child[swap_2_index] = temp
    
    return child 


def generate_initial_pop(pop_size):
    """Get weights for different moves types for whole population

    Args:
        pop_size (int): The number of individuals in the population

    Returns:
        list(tuple(float)): Individuals in population
    """
    P = []
    for i in range(pop_size):
        # Evaluation scores for each contingency
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
        
    return P
    
    
def genetic(max_iters, pop_size):
    """Runs the genetic algorithm

    Args:
        max_iters (int): Max number of generations
        pop_size (int): Population size
    
    Returns:
        list(int): Fittest individual's weights
    """
    P = generate_initial_pop(pop_size)
    fittest_pop = []
    fittest = P[0]
    fittest_fitness = calc_fitness(fittest)
    population_fitness = calc_overall_fitness(P)
    # iterate through max_it times
    for _ in range(max_iters):
        newP = []
        for i in range(pop_size):
            X = get_parent(P, population_fitness)
            Y = get_parent(P, population_fitness)
            Z = reproduce(X, Y)
            if randint(1, 100) > 95:
                Z = mutate(Z)
            Z_fitness = calc_fitness(Z)
            if Z_fitness >= 5:
                fittest_pop.append(Z)
                write(str(Z))
            if Z_fitness > fittest_fitness:
                fittest_fitness = Z_fitness
                fittest = Z
                print(fittest)
            newP.append(Z)
        P = newP
        print('Fittest',fittest)
    fittest = co_evolve()
    return fittest

def co_evolve():
    file = open("./Data/genetic-fit","r")
    P = []
    for line in file:
        individual = line.strip()
        individual = individual.strip("[")
        individual = individual.strip("]")
        individual = [float(i) for i in individual.split(",")]
        P.append(individual)
    while len(P) > 1:
        newP = []
        i = 0
        while i < len(P) - 1:
            print(f"Round {i//2}")
            _, p1score, _, p2score = backgammon(25, "GENETIC", P[i], "GENETIC",P[i+1])
            print(p1score, p2score)
            if p1score > p2score:
                print("winner",P[i-1])
                newP.append(P[i-1])
            else:
                newP.append(P[i])
                print("winner",P[i])
            if i+2 >= len(P):
                newP.append(P[i+1])
            i += 2
        P = newP
    return P[0]


print(genetic(50, 100))
# [0.05574364059282633, 7.0, 0.0, 10.0, 25.0, 15.0, 9.0, 0.8489567804546788, 1.0, 21.0, 13.0, 0.05574364059282633, 0.9241229526062836, 0.7319663189689661, 0.6078305590726676, 0.4989619525836282, 0.05420801597119018, 0.689125038295994]
# [12.0, 1.0, 24.0, 0.5074018981931335, 26.0, 0.5074018981931335, 1.0, 0.3959427019298083, 0.2525167962024688, 4.0, 12.0, 0.3959427019298083, 0.49552490123236437, 0.646808260145097, 0.2680969214986445, 0.5827895413040567, 0.6235369790046915, 0.2525167962024688]
# [4.0, 24.0, 16.0, 4.0, 5.0, 0.37055557981964604, 0.37055557981964604, 21.0, 5.0, 4.0, 25.0, 0.570412964890068, 0.9669697671618662, 0.05674525218346704, 4.0, 0.716577307530303, 0.936915377016144, 0.174253876928524]
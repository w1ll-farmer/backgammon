from random import randint, uniform
from main import backgammon
def get_parent(P):
    # Get fitness values of population
    # get threshold value between 0 and sum of total fitness
    # when cumulative fitness is >= threshold, select individual
    # return
    pass

def reproduce(mother, father):
    crossover_point = randint(0,len(mother)-1)
    child1 = mother[:crossover_point] + father[crossover_point:]
    child2 = father[:crossover_point] + mother[crossover_point:]
    children = [child1, child2]
    return max(children, key=lambda x: calc_fitness(x))


def calc_fitness(individual):
    fitness = backgammon(1, "GENETIC",individual, "GREEDY")[1]
    return fitness


def calc_overall_fitness(P):
    # call calc_fitness for each individual in P
    # return 
    pass


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
        list(list(float)): Individuals in population
    """
    P = []
    for _ in pop_size:
        # weights for: bearing off, hitting, anchoring, exposing
        bear = uniform(0,1)
        hit = uniform(0, 1-bear)
        anchor = uniform(0,1-(bear+hit))
        expose = uniform(0, 1- (bear+hit+anchor))
        other = 0
        if bear + hit + anchor + expose < 1:
            other = 1 -(bear + hit + anchor + expose)
        P.append([bear, hit, anchor, expose, other])
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
    fittest = P[0]
    fittest_fitness = calc_fitness(fittest)
    # iterate through max_it times
    for _ in range(max_iters):
        newP = []
        for i in range(pop_size):
            X = get_parent(P)
            Y = get_parent(P)
            Z = reproduce(X, Y)
            if randint(1, 100) > 95:
                Z = mutate(Z)
            Z_fitness = calc_fitness(Z)
            if Z_fitness > fittest_fitness:
                fittest_fitness = Z_fitness
                fittest = Z
            newP.append(Z)
        P = newP
    return fittest
    
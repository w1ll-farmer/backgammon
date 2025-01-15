from random import randint, uniform
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
    
    # fitness = backgammon(1, "GENETIC",individual, "GREEDY")[1]
    pass
    # return fitness


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
    
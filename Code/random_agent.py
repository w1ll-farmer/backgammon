from random import randint
def generate_random_move():
    return (randint(0,27), randint(0,27))




"""
Select a completely random start point and dice roll
 - Irrespective of whether that is the roll thrown
outward loop, iterating the number of dice they have (2 or 4)
 - Check if the resulting move exists within legal moves, otherwise regen

"""
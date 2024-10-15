from random import randint
def roll():
    """Simulates the rolling of 2 dice
    
    Returns:
        int: The value of the number rolled on dice 1
        int: The value of the number rolled on dice 2
    """
    return randint(1, 6), randint(1, 6)

def is_blot(dest, opp_colour):
    """Checks if the destination is a blot (if it contains a tile that can be hit)

    Args:
        dest (str): The piece(s) on that point.
        opp_colour (str): The first character of the opponents tile colour

    Returns:
        bool: Whether or not the destination point is a blot
    """
    # If the checker can be hit then it will be a single character, either b or w
    # Which are the two possible options of opp_colour, so a simple == check suffices
    return dest == opp_colour
    
    
def is_double(die1, die2):
    return die1 == die2


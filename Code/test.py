# from main import *
from turn import *
from random_agent import *
from time import sleep
import time
def test_double_random():
    moves2, boards2 = get_valid_moves(-1, board, [1,1])
    move = []
    attempts = 0
    
    while move not in moves2 and attempts < 200000:
        move = []
        for i in range(1+is_double([1,1])):
            move.append(generate_random_move())
            move.append(generate_random_move())
        attempts += 1
        
    if move not in moves2:
        move = moves2[randint(0, len(moves2))]
    print(move)
    
# test_double_random()    
""" 
377,801,998,336 possible start-end pairs when rolling a double
614,656 possible start-end pairs when dice are not equal
"""

import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen settings
screen = pygame.display.set_mode((200, 100))
pygame.display.set_caption("Rounded Rectangle")

# Colors
BLUE = (52, 152, 219)  # Color for the box

# Rounded rectangle function
def draw_rounded_rect(surface, color, rect, corner_radius):
    # Unpack rectangle dimensions
    x, y, width, height = rect
    
    # Draw rectangle without corners
    pygame.draw.rect(surface, color, (x + corner_radius, y, width - 2 * corner_radius, height))
    pygame.draw.rect(surface, color, (x, y + corner_radius, width, height - 2 * corner_radius))
    
    # Draw circles in each corner
    pygame.draw.circle(surface, color, (x + corner_radius, y + corner_radius), corner_radius)
    pygame.draw.circle(surface, color, (x + width - corner_radius, y + corner_radius), corner_radius)
    pygame.draw.circle(surface, color, (x + corner_radius, y + height - corner_radius), corner_radius)
    pygame.draw.circle(surface, color, (x + width - corner_radius, y + height - corner_radius), corner_radius)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill screen with white background
    screen.fill((255, 255, 255))
    
    # Draw rounded rectangle
    rect = pygame.Rect(70, 38, 60, 24)  # Position and size of the box
    draw_rounded_rect(screen, BLUE, rect, 4)  # Call the function with radius 16px
    
    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
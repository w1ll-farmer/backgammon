import pygame
from constants import *

class Background: #creates a background
    def __init__(self,backgroundImage):
        # super().__init__()
        # Sets background to passed in image
        self.backgroundImage = pygame.image.load(backgroundImage)
        self.backgroundImage = pygame.transform.scale(self.backgroundImage, (800, 700))
        # Creates a rectangle around it  for co-ordinates
        self.backgroundRect = self.backgroundImage.get_rect()
        
        # Sets X co-ordinates
        self.backgroundX1 = 0 
        # Sets Y co-ordinates
        self.backgroundY1 = (SCREEN_HEIGHT-self.backgroundRect.height)//4 
        # self.backgroundY1 = 0
        
    def render(self): #Renders in the background
        window.blit(self.backgroundImage, (self.backgroundX1, self.backgroundY1))
        
    def update(self,backgroundImage): #Updates the background image
        self.backgroundImage = pygame.image.load(backgroundImage)
        self.backgroundRect = self.backgroundImage.get_rect()

class Shape: #Same as box but takes on an image instead of a colour
    def __init__(self,image,x,y, width=60, height=48):
        self.image = pygame.image.load(image)
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(center = (x,y))
        self.font = pygame.font.SysFont('Calibri',20)
    def move(self,window,x,y): #moves Shape
        self.rect=self.image.get_rect(center=(x,y))
    def draw(self,window): #displays Shape
        window.blit(self.image,self.rect)
    def addText(self,window,text,colour):
        text_surface = self.font.render(text, True, colour)
        
        # Get the text's rect
        text_rect = text_surface.get_rect()

        # Center the text inside the Shape's rect
        text_rect.center = self.rect.center
        
        # Blit the text onto the window
        window.blit(text_surface, text_rect)
import pygame
import numpy as np

class Pixel:
	def __init__(self, pos, scale, threshold = None, **kwargs):
		self.pos = pos
		self.scale = scale
		self.threshold = threshold
		self.color = (255,255,255)
		self.neuron_to_pixel = kwargs.get('neuron_to_pixel', False)
		if self.neuron_to_pixel:
			self.neuron = kwargs.get('neuron', None)
			
	def update(self, *args):
		colorval = 0
		if(self.neuron_to_pixel):
			if self.threshold == None:
				colorval = 255-self.neuron.get_val()
			else:
				colorval = 255-255*(self.neuron.get_val() >= self.threshold)
		else:
			colorval = 255 - args[0]*255
		self.color = (colorval, colorval, colorval)
	
	def draw(self, screen):
		pygame.draw.rect(screen, self.color, [self.pos[0], self.pos[1], self.scale, self.scale])

class PixelGrid:
	def __init__(self, grid, threshold = None, **kwargs):
		self.neuron_to_pixel = kwargs.get('neuron_to_pixel', False)
		if not self.neuron_to_pixel:
			self.screen_width = kwargs.get('screen_width', -1)
		self.pixels = [[None for c in range(len(grid[0]))] for r in range(len(grid))]
		for i in range(len(self.pixels)):
			for j in range(len(self.pixels[0])):
				if self.neuron_to_pixel:
					neuron = grid[i][j]
					# Top left corner
					pixpos = neuron.pos - neuron.scale*neuron.unit_scale
					pixscale = neuron.scale*neuron.unit_scale*2 + neuron.scale*10
					self.pixels[i][j] = Pixel(pixpos, pixscale, threshold, neuron_to_pixel = True, neuron = neuron)
				else:
					width = self.screen_width
					pixpos = np.array([i*width//len(self.pixels), j*width//len(self.pixels[0])])
					pixscale = width/len(self.pixels) - 2
					self.pixels[i][j] = Pixel(pixpos, pixscale, threshold)
	
	def update(self, *args):
		if self.neuron_to_pixel:
			for row in self.pixels:
				for pixel in row:
					pixel.update()
		else:
			grid = args[0]
			for j in range(len(self.pixels[0])):
				for i in range(len(self.pixels)):
					self.pixels[i][j].update(grid[j][i])
				
		
	def draw(self, screen):
		screen.fill((255,255,255))
		for row in self.pixels:
			for pixel in row:
				pixel.draw(screen)
				
				
				
				
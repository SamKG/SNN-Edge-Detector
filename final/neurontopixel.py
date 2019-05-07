import pygame

class Pixel:
	def __init__(self, pos, scale, neuron, threshold = None):
		self.pos = pos
		self.scale = scale
		self.neuron = neuron
		self.threshold = threshold
		self.color = (255,255,255)
	def update(self):
		colorval = 0
		if(self.threshold == None):
			colorval = 255-self.neuron.get_val()
		else:
			colorval = 255-255*(self.neuron.get_val() >= self.threshold)
		self.color = (colorval, colorval, colorval)
	
	def draw(self, screen):
		pygame.draw.rect(screen, self.color, [self.pos[0], self.pos[1], self.scale, self.scale])

class PixelGrid:
	def __init__(self, neurongrid, threshold = None):
		self.pixels = [[None for c in range(len(neurongrid[0]))] for r in range(len(neurongrid))]
		for i in range(len(self.pixels)):
			for j in range(len(self.pixels[0])):
				# Top left corner
				neuron = neurongrid[i][j]
				pixpos = neuron.pos - neuron.scale*neuron.unit_scale
				pixscale = neuron.scale*neuron.unit_scale*2 + neuron.scale*10
				self.pixels[i][j] = Pixel(pixpos, pixscale, neuron, threshold)
	
	def update(self):
		for row in self.pixels:
			for pixel in row:
				pixel.update()
		
	def draw(self, screen):
		screen.fill((255,255,255))
		for row in self.pixels:
			for pixel in row:
				pixel.draw(screen)
				
				
				
				
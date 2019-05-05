import pygame
from numpy import array as nparray
from neuron import Neuron

class Frame:
	def __init__(self, pos, scale_x, scale_y):
		self.pos = pos
		self.scale_x = scale_x
		self.scale_y = scale_y
	def move_frame(self, pos):
		self.pos = pos
	def scale_frame_x(self, scale_x):
		self.scale_x = self.scale_x
	def scale_frame_y(self, scale_y):
		self.scale_y = self.scale_y
	def scale_frame(scale_x, scale_y):
		self.scale_x = scale_x
		self.scale_y = scale_y
		

class DynamicPlot(Frame):
	def __init__(self, pos, scale_x, scale_y, object = None, var_interest = None):
		super().__init__(pos, scale_x, scale_y)
		self.object = object
		self.object_type = None
		self.fix_length = -1
		if(self.object != None):
			self.object_type = object.__class__.__name__
			self.fix_length = object.fix_length
		self.var_interest = var_interest
	def set_var_interest(self, var_interest):
		self.var_interest = var_interest
	def set_object(self, object, var_interest):
		self.object = object
		self.object_type = object.__class__.__name__
		self.fix_length = object.fix_length
		self.var_interest = var_interest	
	def x_axis(self):
		return self.object.times
	def y_axis(self):
		if self.object_type == 'SynapseReader':
			if self.var_interest == 'current':
				return self.object.Is
		elif self.object_type == 'NeuronReader':
			if self.var_interest == 'voltage':
				return self.object.vs
			elif self.var_interest == 'spikes':
				return self.object.spikes
	def draw(self, screen):
		pad = 20
		pygame.draw.rect(screen, (255,255,255), [self.pos[0], self.pos[1], self.scale_x, self.scale_y])
		ratio_x = self.scale_x/screen.get_width();
		ratio_y = self.scale_y/screen.get_height();
		
		x_axis_y = int(self.pos[1] + self.scale_y - pad*ratio_y)
		y_axis_x = int(self.pos[0] + pad*ratio_x)
		
		# Draw the x axis
		pygame.draw.aaline(screen, (0, 0, 0), (int(self.pos[0] + pad*ratio_x), x_axis_y),
								(int(self.pos[0] + self.scale_x - pad*ratio_x), x_axis_y))
		
		# Draw the y axis
		pygame.draw.aaline(screen, (0, 0, 0), (y_axis_x, int(self.pos[1] + pad*ratio_y)),
								(y_axis_x, int(self.pos[1] + self.scale_y - pad*ratio_y)))
		
		x_ax = self.x_axis()
		y_ax = self.y_axis()
		
					
		num_xs = len(x_ax)
		num_ys = len(y_ax)
		
		if num_xs > 0 and num_ys > 0:
		
			def normalize(value, extents):
				if extents[0] == extents[1]:
					return 1
				return (value - extents[0])/(extents[1]-extents[0])
				
			x_ax_extents = [min(x_ax),max(x_ax)]
			
			if self.var_interest != 'spikes':
				y_ax_extents = [min(y_ax),max(y_ax)]
			else:
				y_ax_extents = [0,1]
			
			x_ax_plot = [normalize(x, x_ax_extents)*(self.scale_x - 2*pad*ratio_x) for x in x_ax]
			y_ax_plot = [normalize(y, y_ax_extents)*(self.scale_y - 2*pad*ratio_y) for y in y_ax]
			
			dtick_x = (self.scale_x - 2*pad*ratio_x)/num_xs
			dtick_y = (self.scale_y - 2*pad*ratio_y)/num_ys
			ticklen = 5
			for i in range(num_xs):
				pos_tick_x = nparray([dtick_x*i + pad*ratio_x, x_axis_y])
				pos_tick_y = nparray([y_axis_x, dtick_y*i + pad*ratio_y])
				tick_x_offset = nparray([0, ticklen*ratio_y])
				tick_y_offset = nparray([ticklen*ratio_x, 0])
				tick_x_extents = [pos_tick_x - tick_x_offset, pos_tick_x + tick_x_offset]
				tick_y_extents = [pos_tick_y - tick_y_offset, pos_tick_y + tick_y_offset]
				tick_x_extents = [[int(e[0]),int(e[1])] for e in tick_x_extents]
				tick_y_extents = [[int(e[0]),int(e[1])] for e in tick_y_extents]
				pygame.draw.aaline(screen, (0,0,0),  tick_x_extents[0], tick_x_extents[1])
				pygame.draw.aaline(screen, (0,0,0),  tick_y_extents[0], tick_y_extents[1])
				
				if self.var_interest == 'spikes':
					pygame.draw.aaline(screen, (0,0,0), (x_ax_plot[i] + self.pos[0], x_axis_y),
										(x_ax_plot[i] + self.pos[0], y_ax_plot[i] + self.pos[1]))
				elif i < num_xs-1:
					pygame.draw.aaline(screen, (0,0,0), (x_ax_plot[i] + self.pos[0], y_ax_plot[i] + self.pos[1]),
										(x_ax_plot[i+1] + self.pos[0], y_ax_plot[i+1] + self.pos[1]))

	
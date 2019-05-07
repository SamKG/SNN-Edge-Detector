import pygame
from neuron import Neuron
import numpy as np

class NeuronG(Neuron):
	def __init__(self, pos, v_r = 0, R_m = 1, tau = 1, threshold = 0.2, scale = 1, **kwargs):
		super().__init__(v_r, R_m, tau, threshold, **kwargs)
		self.pos = np.array(pos)
		self.scale = scale
		self.unit_scale = 20
		val = int(self.v / self.threshold)
		self.custom_color = kwargs.get('custom_color', None)
		self.color_by_rate = kwargs.get('color_by_rate', True)
		if self.custom_color is None:
			self.color = (val,0,255-val)
		else:
			self.color = self.custom_color(val)
	
	def get_val(self):
		if not self.color_by_rate:
			val = int(vout / self.threshold * 255)
		else:
			val = int(self.firing_rate * 20 * 255)
		if(val < 0):
			val = 0
		elif(val > 255):
			val = 255
		return val
	
	def update(self, dt, I_inj = 0, learn = False):
		vout = super().update(dt, I_inj, learn)
		if not self.color_by_rate:
			val = int(vout / self.threshold * 255)
			if(val < 0):
				val = 0
			if(val == 255):
				self.color = (255, 255, 0)
			else:
				if self.custom_color is None:
					self.color = (val,0,255-val)
				else:
					self.color = self.custom_color(val)
		else:
			val = int(self.firing_rate*20*255)
			if val > 255:
				val = 255
			if(val < 0):
				val = 0
			if self.custom_color is None:
					self.color = (val,0,255-val)
			else:
				self.color = self.custom_color(val)
		
		return vout
	
	def draw_synapses(self, screen):
		maxI = 2
		for syn in self.syns:
			synval = syn.I
			if synval > maxI:
				synval = maxI
			elif synval < 0:
				synval = 0
			val = int(synval / maxI * 255)
			color = (val,0,255-val)
			pygame.draw.aaline(screen, color, (int(syn.n_pre.pos[0]), int(syn.n_pre.pos[1])),
								(int(syn.n_post.pos[0]), int(syn.n_post.pos[1])))
	
	def draw_neuron(self, screen):
		pygame.draw.circle(screen, self.color, (int(self.pos[0]), int(self.pos[1])), 
							int(self.unit_scale*self.scale))
		
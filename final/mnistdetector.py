import pygame
import math
import numpy as np
import random
import timemodule
from neurongraphics import NeuronG
from mnist_loader import *

pygame.init()

nsize = 855
size = (nsize,nsize)
screen = pygame.display.set_mode(size)

gclock = pygame.time.Clock()

BLACK = (0,0,0)

done = False

dt = 0.01

nclock = timemodule.Clock(dt)

allimages = get_numpy_array()
imgindex = 0
currimg = allimages[imgindex]
print(currimg)

neurongrid = []
'''
scale = 0.55
spacing = 45 * scale
neuroncols = int(float(size[0])/spacing)
neuronrows = int(float(size[1])/spacing)
'''
nneurons = 28
neuroncols = nneurons
neuronrows = nneurons
spacing = 42
scale = int(float(nsize)/nneurons)

for i in range(1, neuronrows):
	row = []
	for j in range(1, neuroncols):
		row.append(NeuronG((j*spacing,i*spacing), scale = scale, isinput = True))
	neurongrid.append(row)

custom_color = lambda val : (val, 255-val, 0)

# Bipolar cells: On center off surround and on center off surround
oncoffs = []
offcons = []
for i in range(1, neuronrows-2):
	oncoffsrow = []
	offconsrow = []
	for j in range(1, neuroncols-2):
		newoncoffs = NeuronG(neurongrid[i][j].pos+(5,5), scale = scale, color = custom_color)
		oncoffsrow.append(newoncoffs)
		newoffcons = NeuronG(neurongrid[i][j].pos+(5,5), scale = scale, color = custom_color)
		offconsrow.append(newoffcons)
		# On center
		newoncoffs.add_syn(neurongrid[i][j],
								w_init = 1, tau = 2, sign=1)
		# Off center
		newoffcons.add_syn(neurongrid[i][j],
							w_init = 0.6, tau = 1, sign=-1)
		
		rotation_1 = 1 + 1j
		rotation_2 = 1
		for _ in range(4):
			curr_i = int(rotation_1.imag) + i
			curr_j = int(rotation_1.real) + j
			# Off surround
			try:
				newoncoffs.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.3, sign=-1)
			except Exception as e:
				print("ngrid rows = ", len(neurongrid))
				print("ngrid cols = ", len(neurongrid[0]))
				print("i = ", i, "j = ", j, "curr_i = ", curr_i, "curr_j = ", curr_j)
				raise(e)
			# On surround
			try:
				newoffcons.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.3, sign=1)
			except Exception as e:
				print("ngrid rows = ", len(neurongrid))
				print("ngrid cols = ", len(neurongrid[0]))
				print("i = ", i, "j = ", j, "curr_i = ", curr_i, "curr_j = ", curr_j)
				raise(e)					
			curr_i = int(rotation_2.imag) + i
			curr_j = int(rotation_2.real) + j
			# Off surround
			try:
				newoncoffs.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.3, sign=-1)
			except Exception as e:
				print("ngrid rows = ", len(neurongrid))
				print("ngrid cols = ", len(neurongrid[0]))
				print("i = ", i, "j = ", j, "curr_i = ", curr_i, "curr_j = ", curr_j)
				raise(e)	
			# On surround
			try:
				newoffcons.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.3, sign=1)
			except Exception as e:
				print("ngrid rows = ", len(neurongrid))
				print("ngrid cols = ", len(neurongrid[0]))
				print("i = ", i, "j = ", j, "curr_i = ", curr_i, "curr_j = ", curr_j)
				raise(e)	
			rotation_1 *= 1j
			rotation_2 *= 1j
			
	oncoffs.append(oncoffsrow)
	offcons.append(offconsrow)
	
	# Ganglion cells
	receptivefield = []
	receptivefieldrow = []
	for i in range(1,len(oncoffs[0])-1):
		currloc = oncoffs[int(len(oncoffs)/2)][i]
		newrc = NeuronG((currloc.pos), scale = scale, color = custom_color)
		receptivefieldrow.append(newrc)
		for j in range(len(oncoffs)):
			newrc.add_syn(oncoffs[j][i], winit = 1, tau = 2)
			newrc.add_syn(offcons[j][i-1], winit = 0.05, tau = 0.5, sign = -1)
			newrc.add_syn(offcons[j][i+1], winit = 0.05, tau = 0.5, sign = -1)
		
	receptivefield.append(receptivefieldrow)	
		
def draw_grid_neurons(neurongrid):
	for nrow in neurongrid:
		for neuron in nrow:
			neuron.draw_neuron(screen)

def draw_grid_synapses(neurongrid):
	for nrow in neurongrid:
		for neuron in nrow:
			neuron.draw_synapses(screen)

def update_grid_neurons(neurongrid):
	for nrow in neurongrid:
		for neuron in nrow:
			neuron.update(nclock.dt)

draw = 0
fc = 0
currline = 0
while not done:
	pressed = False
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			done = True
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_SPACE:
				draw = (draw+1)%4
			if event.key == pygame.K_LEFT:
				imgindex -= 1
			if event.key == pygame.K_RIGHT:
				imgindex += 1
		
	if currline < 0:
		imgindex = 0
	if imgindex > neuroncols-1:
		imgindex = neuroncols-1
	
	currimg = allimages[imgindex]
				
	
	screen.fill(BLACK)
	
	currtime = nclock.get_time()
	
	if draw == 0:
		draw_grid_neurons(neurongrid)
		draw_grid_synapses(neurongrid)
	
	if draw == 1:
		draw_grid_neurons(offcons)
		draw_grid_synapses(offcons)
	
	if draw == 2:
		draw_grid_neurons(oncoffs)
		draw_grid_synapses(oncoffs)
	
	if draw == 3:
		draw_grid_synapses(receptivefield)
		draw_grid_neurons(receptivefield)
			
	for i in range(0, neuronrows):
		for j in range(0, neuroncols):
			neurongrid[i][j].update(nclock.dt, I_inj = 20*currimg[i][j])
	
	update_grid_neurons(oncoffs)
	update_grid_neurons(offcons)
	
	update_grid_neurons(receptivefield)
	
	nclock.tick()
	fc += 1
	
	pygame.display.flip()
	gclock.tick(20)

pygame.quit()
import pygame
import math
import numpy as np
import random
import timemodule
from neurongraphics import NeuronG
from mnist_loader import *

def within_bounds(x, x_l, x_r):
	return x >= x_l and x <= x_r

pygame.init()

DRAW_NEURONS = True
BLOCK_SIZE = 3

nsize = 855
size = (nsize,nsize)
screen = pygame.display.set_mode(size)

gclock = pygame.time.Clock()

BLACK = (0,0,0)

done = False

dt = 0.01
timescale = 4
newtimestep = dt*timescale

nclock = timemodule.Clock(dt)

allimages = get_numpy_array()
imgindex = 0
currimg = allimages[imgindex]

neurongrid = []
nneurons = 28
neuroncols = nneurons
neuronrows = nneurons

spacing = 29.5
scalefactor = 0.8
scale = 20/(nsize/float(spacing))*scalefactor
print(scale)
print(14.5/20)

# photoreceptive layer
for i in range(0, neuronrows):
	row = []
	for j in range(0, neuroncols):
		row.append(NeuronG(((j+1)*spacing,(i+1)*spacing), scale = scale, isinput = True))
	neurongrid.append(row)

custom_color = lambda val : (val, 255-val, 0)

# Bipolar cells: On center off surround and on center off surround
oncoffs = []
offcons = []
for i in range(0, neuronrows):
	oncoffsrow = []
	offconsrow = []
	for j in range(0, neuroncols):
		newoncoffs = NeuronG(neurongrid[i][j].pos+(5,5), scale = scale, custom_color = custom_color)
		oncoffsrow.append(newoncoffs)
		newoffcons = NeuronG(neurongrid[i][j].pos+(5,5), scale = scale, custom_color = custom_color)
		offconsrow.append(newoffcons)
		# On center
		newoncoffs.add_syn(neurongrid[i][j],
								w_init = 1, tau = 2, sign=1)
		# Off center
		newoffcons.add_syn(neurongrid[i][j],
							w_init = 1, tau = 2, sign=-1)
		
		rotation_1 = 1 + 1j
		rotation_2 = 1
		for _ in range(4):
			curr_i = int(rotation_1.imag) + i
			curr_j = int(rotation_1.real) + j
			if within_bounds(curr_i, 0, neuronrows-1) and within_bounds(curr_j, 0, neuroncols-1):
				# Off surround
				newoncoffs.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.2, sign=-1)
				# On surround
				newoffcons.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.25, sign=1)				
			curr_i = int(rotation_2.imag) + i
			curr_j = int(rotation_2.real) + j
			if within_bounds(curr_i, 0, neuronrows-1) and within_bounds(curr_j, 0, neuroncols-1):
				# Off surround
				newoncoffs.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.2, sign=-1)
				# On surround
				newoffcons.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.25, sign=1)
				rotation_1 *= 1j
				rotation_2 *= 1j
			
	oncoffs.append(oncoffsrow)
	offcons.append(offconsrow)

# Line detecting ganglion cells
# each neuron is responsible for detecting its own 3x3 block of on-center off-surround cells surrounding the neuron


line_detectors = []
for i in range(0,len(oncoffs[0])):
	row = []
	for j in range(0,len(oncoffs[0])):
		new_neuron = NeuronG(pos = oncoffs[i][j].pos , scale = scale, custom_color = custom_color)
		row.append(new_neuron)
		top_left_i = i - BLOCK_SIZE//2
		top_left_j = j - BLOCK_SIZE//2
		for bdx in range(0,BLOCK_SIZE*BLOCK_SIZE):
			tmp_i = top_left_i + bdx//BLOCK_SIZE
			tmp_j = top_left_j + (bdx%BLOCK_SIZE)
			#print(tmp_i,tmp_j,top_left_i,top_left_j)
			if (tmp_i >= 0 and tmp_i < len(oncoffs[0])) and (tmp_j >= 0 and tmp_j < len(oncoffs[0])):
				new_neuron.add_syn(oncoffs[tmp_i][tmp_j], winit = 1, tau = 2)
				new_neuron.add_syn(offcons[tmp_i][tmp_j], winit = 1, tau = 2)
	line_detectors.append(row)

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
			if event.key == pygame.K_BACKSPACE:
				draw = (draw-1)%4*(not draw-1 <= 0)
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
	
	if DRAW_NEURONS:
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
			draw_grid_synapses(line_detectors)
			draw_grid_neurons(line_detectors)
	
	this_time = 0
	while this_time < newtimestep:
		this_time += dt
		
		currtime = nclock.get_time()
		
		for i in range(0, neuronrows):
			for j in range(0, neuroncols):
				neurongrid[i][j].update(nclock.dt, I_inj = 10*currimg[i][j])
		
		update_grid_neurons(oncoffs)
		update_grid_neurons(offcons)
		update_grid_neurons(line_detectors)
		nclock.tick()
		
	fc += 1
	
	pygame.display.flip()
	gclock.tick(20)

pygame.quit()
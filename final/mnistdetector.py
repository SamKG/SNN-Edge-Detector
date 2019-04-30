import pygame
import os
import subprocess
import math
import numpy as np
import random
import timemodule
from neurongraphics import NeuronG
import mnist_loader

# Color constant for black
BLACK = (0,0,0)
WHITE = (255,255,255)

def within_bounds(x, x_l, x_r):
	return x >= x_l and x <= x_r

class Label:
	def __init__(self, init_idx=0):
		self.font = pygame.font.SysFont("Segoe UI", 65)
		self.labels = {0:"Photoreceptors", 1:"Off-Center-On-Surround",
					2:"On-Center-Off-Surround", 3:"Ganglion", 4:"Ganglion Vert/Horiz", 5:"Ganglion Diags"}
		self.anim_dur = 0
		self.anim_curr = 0
		self.currlabel = self.font.render(self.labels[init_idx], True, WHITE)
		self.alphasurf = pygame.Surface(self.currlabel.get_size(),
										pygame.SRCALPHA)
		self.alphasurf.fill((255,255,255,0))
		self.currlabel.blit(self.alphasurf, (0,0), 
							special_flags=pygame.BLEND_RGBA_MULT)
	
	def update_label(self, idx):
		self.currlabel = self.font.render(self.labels[idx], True, WHITE)
		self.alphasurf = pygame.Surface(self.currlabel.get_size(),
										pygame.SRCALPHA)
	
	def anim_start(self, dur):
		self.anim_dur = dur
		self.anim_curr = 0
		self.alphasurf.fill((255,255,255,255-int(255*self.anim_curr/self.anim_dur)))
		self.currlabel.blit(self.alphasurf, (0,0), 
							special_flags=pygame.BLEND_RGBA_MULT)
	
	def anim_update(self):
		if self.anim_curr < self.anim_dur:
			self.anim_curr += 1
			self.alphasurf.fill((255,255,255,255-int(255*self.anim_curr/self.anim_dur)))
			self.currlabel.blit(self.alphasurf, (0,0), 
								special_flags=pygame.BLEND_RGBA_MULT)
	
	def draw(self, screen, pos):
		screen.blit(self.currlabel, pos)

pygame.init()

# Making the directory for storing frames
frame_dir = "frames"
try:
	os.makedirs(frame_dir)
except FileExistsError:
	filestoremove = [os.path.abspath(os.path.join(frame_dir, f)) 
		for f in os.listdir(frame_dir) 
		if os.path.isfile(os.path.join(frame_dir, f))]
	for f in filestoremove:
		os.remove(f)

curr_recording_idx = 0	
# Making the directory for the video
record_dir = "recordings"
try:
	os.makedirs(record_dir)
except FileExistsError:
	videofiles = [f for f in os.listdir(record_dir)
					if (os.path.isfile(os.path.join(record_dir, f)) and f != ".gitignore")]
	if not (not videofiles):
		vfnums = [int(os.path.splitext(vf)[0]) for vf in videofiles]
		curr_recording_idx = max(vfnums)+1

# Constant for whether to draw
DRAW_NEURONS = True

# Defines the screen
nsize = 856
size = (nsize,nsize)
screen = pygame.display.set_mode(size)

# Creates the game clock
gclock = pygame.time.Clock()

done = False

# Time stuff for neuron updating
dt = 0.01
timescale = 4
newtimestep = dt*timescale

nclock = timemodule.Clock(dt)

# Getting the MNIST images for the photoreceptive layer
allimages = mnist_loader.get_numpy_array()
imgindex = 0
currimg = allimages[imgindex]

# Defining constants for our neuron grid size and neuron scale and spacing
neurongrid = []
nneurons = 28
neuroncols = nneurons
neuronrows = nneurons
spacing = 29.5
scalefactor = 0.8
scale = 20/(nsize/float(spacing))*scalefactor

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
								w_init = 1, tau = 1, sign=1)
		# Off center
		newoffcons.add_syn(neurongrid[i][j],
							w_init = 1, tau = 1, sign=-1)
		
		rotation_1 = 1 + 1j
		rotation_2 = 1
		for _ in range(4):
			curr_i = int(rotation_1.imag) + i
			curr_j = int(rotation_1.real) + j
			if within_bounds(curr_i, 0, neuronrows-1) and within_bounds(curr_j, 0, neuroncols-1):
				# Off surround
				newoncoffs.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.1, sign=-1)
				# On surround
				newoffcons.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.02, sign=1)				
			curr_i = int(rotation_2.imag) + i
			curr_j = int(rotation_2.real) + j
			if within_bounds(curr_i, 0, neuronrows-1) and within_bounds(curr_j, 0, neuroncols-1):
				# Off surround
				newoncoffs.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.1, sign=-1)
				# On surround
				newoffcons.add_syn(neurongrid[curr_i][curr_j],
									w_init = 0.02, sign=1)
				rotation_1 *= 1j
				rotation_2 *= 1j
			
	oncoffs.append(oncoffsrow)
	offcons.append(offconsrow)

# Constant for line detecting ganglion cells
BLOCK_SIZE = 3	

# Line detecting ganglion cells
# each neuron is responsible for detecting its own 3x3 block of on-center off-surround cells surrounding the neuron
line_detectors = []
line_detectors_vh = []
line_detectors_d = []
for i in range(0,neuronrows):
	row = []
	row_vh = []
	row_d = []
	for j in range(0,neuroncols):
		top_i = i - BLOCK_SIZE//2
		bottom_i = i + BLOCK_SIZE//2
		left_j = j - BLOCK_SIZE//2
		right_j = j + BLOCK_SIZE//2
		pos1 = np.array(oncoffs[i][j].pos)
		pos2 = np.array(oncoffs[top_i][left_j].pos)
		# This neuron will detect vertical and horizontal lines
		vert_horiz = NeuronG(pos = pos1 + (pos2 - pos1)/4, scale = scale, custom_color = custom_color)
		row_vh.append(vert_horiz)
		row.append(vert_horiz)
		if(left_j >= 0 and right_j < neuroncols):
			for tmp_j in range(left_j, right_j+1):
				vert_horiz.add_syn(offcons[i][tmp_j], winit = 0.3, tau = 4)
		if(top_i >= 0 and bottom_i < neuronrows):
			for tmp_i in range(top_i, bottom_i+1):
				vert_horiz.add_syn(offcons[tmp_i][j], winit = 1, tau = 4)
		if(top_i >= 0 and bottom_i < neuronrows and left_j >= 0 and right_j < neuroncols):
			# This neuron will detect diagonal lines
			diags = NeuronG(pos = pos1 + (pos1 - pos2)/2, scale = scale, custom_color = custom_color)
			row.append(diags)
			row_d.append(diags)
			for tmp in range(-(BLOCK_SIZE//2), BLOCK_SIZE//2+1):
				diags.add_syn(offcons[i + tmp][j + tmp], winit = 0.3, tau = 4)
				diags.add_syn(offcons[i - tmp][j + tmp], winit = 1, tau = 4)
		else:
			row_d.append(None)

	line_detectors.append(row)
	line_detectors_vh.append(row_vh)
	line_detectors_d.append(row_d)

# Mapping back to the photoreceptive layer

def draw_grid_neurons(neurongrid):
	for nrow in neurongrid:
		for neuron in nrow:
			if neuron != None:
				neuron.draw_neuron(screen)

def draw_grid_synapses(neurongrid):
	for nrow in neurongrid:
		for neuron in nrow:
			if neuron != None:
				neuron.draw_synapses(screen)

def update_grid_neurons(neurongrid, I_inj = 0):
	for nrow in neurongrid:
		for neuron in nrow:
			if neuron != None:
				neuron.update(nclock.dt, I_inj = I_inj)

draw_type = 0
record = False
saved_frame = 0
fc = 0
framerate = 20

mylabel = Label(draw_type)
mylabelpos = [size[0]/2-mylabel.currlabel.get_size()[0]/2,50]
labelanimlen = 300
mylabel.update_label(draw_type)
mylabel.anim_start(labelanimlen)

num_layers = 6

while not done:
	pressed = False
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			done = True
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_SPACE:
				draw_type = (draw_type+1)%num_layers
				mylabel.update_label(draw_type)
				mylabel.anim_start(labelanimlen)
				mylabelpos = [size[0]/2-mylabel.currlabel.get_size()[0]/2,50]
			if event.key == pygame.K_BACKSPACE:
				draw_type = (draw_type-1)%num_layers
				mylabel.update_label(draw_type)
				mylabel.anim_start(labelanimlen)
				mylabelpos = [size[0]/2-mylabel.currlabel.get_size()[0]/2,50]
			if event.key == pygame.K_LEFT:
				imgindex = (imgindex-1)%neuroncols
			if event.key == pygame.K_RIGHT:
				imgindex = (imgindex+1)%neuroncols
			if event.key == pygame.K_r:
				record = not record
	
	currimg = allimages[imgindex]
				
	
	screen.fill(BLACK)
	
	if DRAW_NEURONS:
		if draw_type == 0:
			draw_grid_neurons(neurongrid)
			draw_grid_synapses(neurongrid)
			mylabel.draw(screen, mylabelpos)
		
		if draw_type == 1:
			draw_grid_neurons(offcons)
			draw_grid_synapses(offcons)
			mylabel.draw(screen, mylabelpos)
		
		if draw_type == 2:
			draw_grid_neurons(oncoffs)
			draw_grid_synapses(oncoffs)
			mylabel.draw(screen, mylabelpos)
		
		if draw_type == 3:
			draw_grid_synapses(line_detectors)
			draw_grid_neurons(line_detectors)
			mylabel.draw(screen, mylabelpos)
		
		if draw_type == 4:
			draw_grid_synapses(line_detectors_vh)
			draw_grid_neurons(line_detectors_vh)
			mylabel.draw(screen, mylabelpos)
		
		if draw_type == 5:
			draw_grid_synapses(line_detectors_d)
			draw_grid_neurons(line_detectors_d)
			mylabel.draw(screen, mylabelpos)
	
	this_time = 0
	while this_time < newtimestep:
		this_time += dt
		
		currtime = nclock.get_time()
		
		for i in range(0, neuronrows):
			for j in range(0, neuroncols):
				neurongrid[i][j].update(nclock.dt, I_inj = 20*currimg[i][j])
		update_grid_neurons(oncoffs, I_inj = 0.5)
		update_grid_neurons(offcons, I_inj = 0.5)
		update_grid_neurons(line_detectors)
		nclock.tick()
	
	line_detectors_d[14][14].color = (255,255,255)
	for synapse in line_detectors_d[14][14].syns:
		synapse.n_pre.color = (255,255,255)
		
	fc += 1
	
	mylabel.anim_update()
	
	if record:
		pygame.image.save(screen, 
			os.path.join(frame_dir,('img%d' % saved_frame)+".png"))
		saved_frame += 1
	pygame.display.flip()
	gclock.tick(framerate)

pygame.quit()

frames_exist = not not [f for f in os.listdir(frame_dir) 
			if os.path.isfile(os.path.join(frame_dir, f))]
if frames_exist:
	inputfilestring = frame_dir + '/' + 'img%d.png'
	outputfilestring = record_dir + '/' + str(curr_recording_idx)+'.mp4'
	ffmpegpath = "ffmpeg"
	subprocess.call([ffmpegpath, '-framerate', str(framerate//4), 
	'-i', inputfilestring, '-crf', str(framerate//4), '-pix_fmt', 'yuv420p', 
	outputfilestring])
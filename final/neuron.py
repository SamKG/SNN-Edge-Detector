import numpy as np
from collections import deque as Queue
import math

t_window = 1.5
t_step = 5

class Synapse:
	def __init__(self, n_pre, n_post, tau = 1, w_init = 0.5, type = None, **kwargs):
		self.n_pre = n_pre
		self.n_post = n_post
		self.w = w_init
		self.I = 0
		self.tau = tau
		self.sign = kwargs.get('sign', 1)
		self.effective = kwargs.get('effective', True)
		if(type == 'hebbian'):
			self.type = 'hebbian'
			self.gamma = kwargs.get('gamma', 0.5)
		elif(type == 'stdp'):
			self.type = 'stdp'
			self.tau_p = kwargs.get('tau_p', 1)
			self.tau_m = kwargs.get('tau_m', 1)
			self.eta_p = kwargs.get('eta_p', 5)
			self.eta_m = kwargs.get('eta_m', 2)
			self.x_j = 0
			self.y_i = 0
			
	def update(self, dt):
		dIdt = (-self.I/self.tau) + self.n_pre.currspike
		self.I = self.I + dt*dIdt
	
	def learn(self, dt):
		if(self.type == 'hebbian'):
			dw = 0
			for _ in range(4):
				dw += 0.25 * self.gamma * (self.n_post.spike_rate * 
							(self.n_pre.spike_rate - self.w*self.n_post.spike_rate))
			self.w = self.w + dw
			
		elif(self.type == 'stdp'):
			if len(self.n_pre.spikes) > 0 and len(self.n_post.spikes) > 0:
				self.x_j = self.x_j + dt*(-self.x_j/self.tau_p) + self.n_pre.currspike
				self.y_i = self.y_i + dt*(-self.y_i/self.tau_m) + self.n_post.currspike
				A_p = (1 - self.w)*self.eta_p
				A_m = self.w*self.eta_m
				dwdt = (A_p*self.n_pre.currspike*self.x_j -
						A_m*self.n_post.currspike*self.y_i)
				self.w = self.w + dt*dwdt
			
	def sout(self):
		return self.sign * self.w * self.I

class Neuron:
	def __init__(self, v_r = 0, R_m = 1, tau = 1, threshold = 0.2, type = None, **kwargs):
		self.v = v_r
		self.v_r = v_r
		self.R_m = R_m
		self.tau = tau
		self.threshold = threshold
		self.type = type
		self.syns = []
		self.spikes = Queue()
		self.spikestoadd = []
		self.currspike = 0
		self.spike_rate = 0
		self.isteacher = kwargs.get('isteacher', False)
		self.isinput = kwargs.get('isinput', False)
	
	def add_syn(self, n_pre, tau = 1, w_init = 0.5, **kwargs):
		syn = Synapse(n_pre, self, tau = tau, w_init = w_init, type = self.type, **kwargs)
		self.syns.append(syn)
		return syn
	
	def remove_syn(self, n_pre):
		for syn in self.syns:
			if syn.n_pre == n_pre:
				self.syns.remove(syn)
	
	def get_syn(self, n_pre):
		for syn in self.syns:
			if syn.n_pre == n_pre:
				return syn
		return None
	
	def I_syn(self):
		# We multiply by effective because we only want the synapses that are "effective"
		# to have their current influence the input current
		I_list = [syn.sout()*syn.effective for syn in self.syns]
		return sum(I_list)
	
	def add_spike(self, val, dt):
		self.currspike = val
		if self.type == 'hebbian':
			global t_step;
			global t_window;
			if len(self.spikestoadd) < int(t_step/dt):
				self.spikestoadd.append(val)
			else:
				for spike in self.spikestoadd:
					self.spikes.append(spike)
					if(len(self.spikes) > t_window/dt):
						self.spikes.popleft()
				self.spikestoadd = []

			self.spike_rate = sum(self.spikes)/(t_window/dt)
		else:
			self.spikes.append(val)
			if(len(self.spikes) > t_window/dt):
						self.spikes.popleft()
	
	# Returns the firing rate of spikes per second
	def get_firing_rate(self):
		if len(self.spikes) > 0:
			return float(sum(self.spikes))/len(self.spikes)
		else:
			return 0
	
	def update(self, dt, I_inj = 0, learn = False):
		if not self.isteacher:
			for syn in self.syns:
				syn.update(dt)
				# If the presynapitc neuron is a teacher, don't change its weights
				# otherwise do.
				if learn and not syn.n_pre.isteacher:
				  syn.learn(dt)
		# If this neuron is a teacher or input, then we just take in the injected current i.e.
		# no synaptic connections
		if self.isteacher or self.isinput:
			I_total = I_inj
		# Otherwise aggregate all the weighted synaptic inputs and injected current
		else:
			I_total = self.I_syn() + I_inj
		if math.isnan(I_total):
			raise Exception("Current is NaN")
		# Leaky integrate and fire dynamics
		dvdt = (-self.v + self.R_m * I_total)/self.tau
		self.v = self.v + dt*dvdt
		if(self.v >= self.threshold):
			self.v = self.v_r
			self.add_spike(1, dt) 
			return self.threshold
		else:
			self.add_spike(0, dt)
			return self.v

class SynapseReader:
	def __init__(self, synapse, fix_length = -1):
		self.synapse = synapse
		self.fix_length = fix_length
		self.times = []
		self.Is = []
		self.ws = []
		
	def read_synapse(self, nclock):
		self.times.append(nclock.get_time())
		self.Is.append(self.synapse.I)
		self.ws.append(self.synapse.w)

		if self.fix_length > 0 and len(self.times) >= self.fix_length:
			self.times = self.times[-self.fix_length:]
			self.Is = self.Is[-self.fix_length:]
			self.ws = self.ws[-self.fix_length:]
		
	def refresh(self):
		self.Is = []
		self.ws = []
			
class NeuronReader:
	def __init__(self, neuron, readsyns = False, fix_length = -1, var_interest = None):
		self.neuron = neuron
		self.readsyns = readsyns
		if self.readsyns:
			self.syns = self.neuron.syns
			self.synreaders = [SynapseReader(syn, fix_length) for syn in self.neuron.syns]
		self.fix_length = fix_length
		self.times = []
		self.vs = []
		self.spikes = []
		
	def change_neuron(self, neuron, readsyns = False):
		self.neuron = neuron
		self.times = []
		self.vs = []
		self.spikes = []
		if self.readsyns:
			self.synreaders = [SynapseReader(syn) for syn in self.neuron.syns]
			
	def update_synapses(self):
		if self.readsyns:
			for syn in self.neuron.syns:
				if syn not in self.syns:
					self.synreaders.append(SynapseReader(syn))
					
	def refresh(self):
		self.vs = []
		self.spikes = []
		self.times = []
		if self.readsyns:
			for synreader in self.synreaders:
				synreader.refresh()
				
	def read_neuron(self, nclock):
		self.times.append(nclock.get_time())
		self.vs.append(self.neuron.v)
		self.spikes.append(self.neuron.currspike)
		if self.fix_length > 0 and len(self.vs) >= self.fix_length:
			self.times = self.times[-self.fix_length:]
			self.vs = self.vs[-self.fix_length:]
			self.spikes = self.spikes[-self.fix_length:]
		if self.readsyns:
			for synreader in self.synreaders:
				synreader.read_synapse(nclock)
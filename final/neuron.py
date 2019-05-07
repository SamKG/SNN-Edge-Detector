import numpy as np
from collections import deque as Queue
import math

t_window = 2
t_step = 5

class Synapse:
	def __init__(self, n_pre, n_post, tau = 1, w_init = 0.5, **kwargs):
		self.n_pre = n_pre
		self.n_post = n_post
		self.w = w_init
		self.I = 0
		self.tau = tau
		self.sign = kwargs.get('sign', 1)
			
	def update(self, dt):
		dIdt = (-self.I/self.tau) + self.n_pre.currspike
		self.I = self.I + dt*dIdt
			
	def sout(self):
		return self.sign * self.w * self.I

class Neuron:
	def __init__(self, v_r = 0, R_m = 1, tau = 1, threshold = 0.2, **kwargs):
		self.v = v_r
		self.v_r = v_r
		self.R_m = R_m
		self.tau = tau
		self.threshold = threshold
		self.syns = []
		self.spikes = Queue()
		self.currspike = 0
		self.firing_rate = 0
		self.isinput = kwargs.get('is_input', False)
		self.compute_firing_rate = kwargs.get('compute_firing_rate', True)
	
	def add_syn(self, n_pre, tau = 1, w_init = 0.5, **kwargs):
		syn = Synapse(n_pre, self, tau = tau, w_init = w_init, **kwargs)
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
		I_list = [syn.sout() for syn in self.syns]
		return sum(I_list)
	
	def add_spike(self, val, dt):
		global t_window
		self.currspike = val
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
		for syn in self.syns:
			syn.update(dt)
		# If this neuron is input, then we just take in the injected current i.e.
		# no synaptic connections
		if self.isinput:
			I_total = I_inj
		# Otherwise aggregate all the weighted synaptic inputs and injected current
		else:
			I_total = self.I_syn() + I_inj
		# Catching when current input is unreasonably high
		if math.isnan(I_total):
			raise Exception("Current is NaN")
		# Leaky integrate and fire dynamics
		dvdt = (-self.v + self.R_m * I_total)/self.tau
		self.v = self.v + dt*dvdt
		if(self.v >= self.threshold):
			self.v = self.v_r
			self.add_spike(1, dt) 
			# Compute the firing rate if desired
			if(self.compute_firing_rate):
				self.firing_rate = self.get_firing_rate()
			return self.threshold
		else:
			self.add_spike(0, dt)
			# Compute the firing rate if desired
			if(self.compute_firing_rate):
				self.firing_rate = self.get_firing_rate()
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
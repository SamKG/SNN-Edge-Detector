from brian2 import *
from brian2tools import *

REFRACTORY_TIME = 0*ms
INTEG_METHOD = 'euler'
D_T = .1*ms # set this for default timestep
defaultclock.dt = D_T

NEURON_ROW_SIZE = 28
NUM_NEURONS = NEURON_ROW_SIZE*NEURON_ROW_SIZE
RECEPTIVE_FIELD = 3
BLOCK_SIZE = 3
tau = 10*ms
R = 1*ohm
C = 1*farad
q = 1
eqs = '''
dv/dt = -(v-(I1 + I2))/tau : volt (unless refractory)
da/dt = -1/tau * a : 1
I1 : volt
I2 : volt
'''

syn_eqs = '''
w:1 
I1_post = w*(a_pre) : volt(summed)
'''
syn_eqs_2 = '''
w:1 
I2_post = w*(a_pre) : volt(summed)
'''
syn_on_pre = 'a+=1'

from brian2 import *
from brian2tools import *

NUM_NEURONS = 28*28
RECEPTIVE_FIELD = 3
BLOCK_SIZE = 3
tau = 10*ms
R = 1*ohm
C = 1*farad
q = 1
eqs = '''
dv/dt = -(v-I)/tau : volt (unless refractory)
da/dt = -1/tau * a : 1
I : volt
'''

syn_eqs = '''
w:1 
I_post = w*(a_pre) (summed)
'''
syn_on_pre = 'a+=1'
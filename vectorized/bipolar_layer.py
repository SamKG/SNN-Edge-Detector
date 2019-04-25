from parameters import *
from brian2 import *
from brian2tools import *

OnCenterOffSurround = NeuronGroup(NUM_NEURONS, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=REFRACTORY_TIME, method=INTEG_METHOD)
OnCenterOffSurround.v = 0*mV # initialize voltages to 0
OnCenterOffSurround.a = 0

OffCenterOnSurround = NeuronGroup(NUM_NEURONS, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=REFRACTORY_TIME, method=INTEG_METHOD)
OffCenterOnSurround.v = 0*mV # initialize voltages to 0
OffCenterOnSurround.a = 0


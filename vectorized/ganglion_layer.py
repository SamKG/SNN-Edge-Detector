from parameters import *
from brian2 import *
from brian2tools import *

GanglionLayer = NeuronGroup(NUM_NEURONS, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=REFRACTORY_TIME, method=INTEG_METHOD)
GanglionLayer.v = 0*mV # initialize voltages to 0
GanglionLayer.a = 0


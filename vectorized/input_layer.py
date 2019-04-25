from parameters import *
from brian2 import *
from brian2tools import *

InputLayer = NeuronGroup(NUM_NEURONS, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method='exact')
InputLayer.v = 0*mV # initialize voltages to 0
InputLayer.a = 0 # init spike traces to 0
# Given a 2d array of input currents, sets the input currents to the input layer
def setInputLayer(input_array):
    curr = 0
    for row in input_array:
        for val in row:
            InputLayer[curr].I = val * mV
            curr+=1


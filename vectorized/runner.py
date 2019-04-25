from input_layer import *
from bipolar_layer import *
from parameters import *

input_oncenter = Synapses(InputLayer,OnCenterOffSurround,model=syn_eqs)
input_offcenter = Synapses(InputLayer,OffCenterOnSurround,model=syn_eqs)


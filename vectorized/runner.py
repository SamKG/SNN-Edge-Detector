from input_layer import *
from bipolar_layer import *
from ganglion_layer import *
from parameters import *
from pathlib import Path

input_oncenter = Synapses(InputLayer,OnCenterOffSurround,model=syn_eqs)
input_offcenter = Synapses(InputLayer,OffCenterOnSurround,model=syn_eqs)
oncenter_ganglion = Synapses(OnCenterOffSurround,GanglionLayer,model=syn_eqs)
offcenter_ganglion = Synapses(OffCenterOnSurround,GanglionLayer,model=syn_eqs)



NETWORK = Network([InputLayer,
OnCenterOffSurround,
OffCenterOnSurround,
GanglionLayer,
input_oncenter,
input_offcenter,
oncenter_ganglion,
offcenter_ganglion])


### ONCENTER LAYER ####
if (Path('stored_states/one.state').is_file()):
    print('LOADING ONE FROM FILE')
    NETWORK.restore('one','stored_states/one.state')
else:
    W_INHIB = .2
    W_EXCIT = 1
    # connect input-oncenter
    for i in range(0,NEURON_ROW_SIZE):
        for j in range(0,NEURON_ROW_SIZE):
            # transform to pos in 1d array
            curr_indx = (i*NEURON_ROW_SIZE) + j 
            i_topleft = i - RECEPTIVE_FIELD//2
            j_topleft = j - RECEPTIVE_FIELD//2
            for d in range(0,RECEPTIVE_FIELD*RECEPTIVE_FIELD):
                # get indices of top left square in receptive field
                tmp_i = i_topleft + d//RECEPTIVE_FIELD 
                tmp_j = j_topleft + (d%RECEPTIVE_FIELD)
                # convert to pos in 1d array (needed for brian2)
                tmp_indx = (tmp_i * NEURON_ROW_SIZE) + tmp_j
                if ( tmp_i >= 0 and tmp_i < NEURON_ROW_SIZE and tmp_j >= 0 and tmp_j < NEURON_ROW_SIZE):
                    #print(curr_indx,tmp_indx,tmp_i,tmp_j)
                    input_oncenter.connect(i=tmp_indx,j=curr_indx) # connect input to neuron
                    if (tmp_i == i and tmp_j == j):
                        input_oncenter.w[tmp_indx:curr_indx] = W_EXCIT # if is center, then it is excitatory
                    else:
                        input_oncenter.w[tmp_indx:curr_indx] = W_INHIB # if is surround, then it is inhibitory
    NETWORK.store('one','stored_states/one.state')
print('ONE LOADED')



### OFFCENTER LAYER ####
if (Path('stored_states/two.state').is_file()):
    print('LOADING TWO FROM FILE')
    NETWORK.restore('two','stored_states/two.state')
else:
    W_INHIB = 1
    W_EXCIT = .1
    # connect input-offcenter
    for i in range(0,NEURON_ROW_SIZE):
        for j in range(0,NEURON_ROW_SIZE):
            # transform to pos in 1d array
            curr_indx = (i*NEURON_ROW_SIZE) + j 
            i_topleft = i - RECEPTIVE_FIELD//2
            j_topleft = j - RECEPTIVE_FIELD//2
            for d in range(0,RECEPTIVE_FIELD*RECEPTIVE_FIELD):
                # get indices of top left square in receptive field
                tmp_i = i_topleft + d//RECEPTIVE_FIELD 
                tmp_j = j_topleft + (d%RECEPTIVE_FIELD)
                # convert to pos in 1d array (needed for brian2)
                tmp_indx = (tmp_i * NEURON_ROW_SIZE) + tmp_j
                if ( tmp_i >= 0 and tmp_i < NEURON_ROW_SIZE and tmp_j >= 0 and tmp_j < NEURON_ROW_SIZE):
                    #print(curr_indx,tmp_indx,tmp_i,tmp_j)
                    input_offcenter.connect(i=tmp_indx,j=curr_indx) # connect input to neuron
                    if (tmp_i == i and tmp_j == j):
                        input_offcenter.w[tmp_indx:curr_indx] = W_INHIB # if is center, then it is inhibitory
                    else:
                        input_offcenter.w[tmp_indx:curr_indx] = W_EXCIT # if is surround, then it is excitatory
    NETWORK.store('two','stored_states/two.state')
print('TWO LOADED')


### GANGLION LAYER ###
if (Path('stored_states/three.state').is_file()):
    print('LOADING THREE FROM FILE')
    NETWORK.restore('three','stored_states/three.state')
else:
    W_EXCIT = 1
    W_INHIB = .1
    #connect to ganglion layer
    for i in range(0,NEURON_ROW_SIZE):
        for j in range(0,NEURON_ROW_SIZE):
            # transform to pos in 1d array
            curr_indx = (i*NEURON_ROW_SIZE) + j 
            i_topleft = i - BLOCK_SIZE//2
            j_topleft = j - BLOCK_SIZE//2
            for d in range(0,BLOCK_SIZE*BLOCK_SIZE):
                # get indices of top left square in receptive field
                tmp_i = i_topleft + d//BLOCK_SIZE 
                tmp_j = j_topleft + (d%BLOCK_SIZE)
                # convert to pos in 1d array (needed for brian2)
                tmp_indx = (tmp_i * NEURON_ROW_SIZE) + tmp_j
                if ( tmp_i >= 0 and tmp_i < NEURON_ROW_SIZE and tmp_j >= 0 and tmp_j < NEURON_ROW_SIZE):
                    #print(curr_indx,tmp_indx,tmp_i,tmp_j)
                    oncenter_ganglion.connect(i=tmp_indx,j=curr_indx) # connect input to neuron
                    if (tmp_i == i and tmp_j == j):
                        oncenter_ganglion.w[tmp_indx:curr_indx] = W_INHIB # if is surround, then it is inhibitory
                    else:
                        oncenter_ganglion.w[tmp_indx:curr_indx] = W_EXCIT # if is center, then it is excitatory
                    offcenter_ganglion.connect(i=tmp_indx,j=curr_indx) # connect input to neuron
                    if (tmp_i == i and tmp_j == j):
                        offcenter_ganglion.w[tmp_indx:curr_indx] = W_INHIB # if is surround, then it is inhibitory
                    else:
                        offcenter_ganglion.w[tmp_indx:curr_indx] = W_EXCIT # if is center, then it is excitatory
    NETWORK.store('three','stored_states/three.state')
print('THREE LOADED')



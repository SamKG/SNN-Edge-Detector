import matplotlib.pyplot as plt
import numpy as np
import math
import random 

HEBBIAN_TERM = .0867
EPOCH_TIME = 20
class InputNeuron:
    IsInput = True
    voltage = 0
    def __init__(self,current=1):
        self.current = current

    def getOutput(self):
        return self.current

    def step(self,timestep=1):
        pass
    def train(self,timestep = 1):
        pass
    def resetEpoch(self):
        pass
    
class Neuron:
    totalTime = 0
    IsInput = False
    epochTime = 0
    def __init__(self,a=.02,b=.2,c=-65,d=2,vt=30):
        self.a = a
        self.b = b
        self.c = c
        self.Vt = vt
        self.voltage = -70 # Resting potential
        self.u = self.voltage * self.b
        self.d = d
        self.totalTime = 0
        self.spikeTimes = []
        self.maxSpikeCount = EPOCH_TIME
        self.preSynaptic = []
        self.tau = 7
        self.numberSpikes = 0

    def spike(self):
        self.spikeTimes = [s for s in self.spikeTimes if self.totalTime - s <= self.maxSpikeCount]
        self.spikeTimes.append(self.totalTime)
        self.numberSpikes += 1
    
    def step(self,timestep=1,input = None):
        if input is None:
            input = self.getInput()
        self.totalTime += timestep
        if self.voltage >= self.Vt: # Fire at will!
           # print("SPIKE",self.Vm,self.v)
            self.voltage = self.c # Reset to resting potential
            self.u = self.u + self.d
            self.spike()
        dv = (0.04*self.voltage*self.voltage) + (5*self.voltage) + 140 - self.u + input        
        self.voltage += dv*timestep
        # This double update thing was in the paper for 'numerical stability' - I'll just leave it here
        dv = (0.04*self.voltage*self.voltage) + (5*self.voltage) + 140 - self.u + input        
        self.voltage += dv*timestep
        du = self.a *( (self.b * (self.voltage)) - self.u)
        self.u += du*timestep
    def resetTimer(self):
        self.totalTime = 0
        self.epochTime = 0

    def getInput(self):
        total = 0
        for n in self.preSynaptic:
            w = n[0]
            pre_syn_neuron = n[1]
            total += w*pre_syn_neuron.getOutput()
        return total
    
    def getOutput(self):
        Isyn = 0
        for t in self.spikeTimes:
            dt = self.totalTime - t
            Isyn += 30/self.tau * math.exp(-1*dt/self.tau)
        return Isyn 

    def getActivity(self):
        if len(self.spikeTimes) == 0:
            return 0 
        if self.spikeTimes[0] == self.totalTime:
            return 1        
        return len(self.spikeTimes)/(self.totalTime - self.spikeTimes[0])
        #return self.numberSpikes / self.epochTime
    
    def resetEpoch(self):
        self.numberSpikes = 0
        self.epochTime = 0

    def connectPreSynaptic(self,neuron,weight = None):
        if weight is None:
            weight = random.uniform(-1/math.sqrt(2),1/math.sqrt(2))
        self.preSynaptic.append([weight,neuron])

    def train(self,timestep = 1):
        sq_sum = 0
        for i in range(0,len(self.preSynaptic)):
            if not self.preSynaptic[i][1].IsInput:
                vi = self.preSynaptic[i][1].getActivity()
                vj = self.getActivity()
                if vi > 0: print("Vi",vi, "Vj",vj)
                wij = self.preSynaptic[i][0]
                dw = HEBBIAN_TERM * (vi*vj - wij*vj*vj)
                #if dw > 0: print("dw",dw,"vi",vi,"vj",vj,n[0],n[0]+dw)
                self.preSynaptic[i][0] += dw*timestep
                sq_sum += self.preSynaptic[i][0]**2
        sq_sum = math.sqrt(sq_sum)
        for i in range(0,len(self.preSynaptic)):
            if not self.preSynaptic[i][1].IsInput:
                self.preSynaptic[i][0] = self.preSynaptic[i][0]/sq_sum

class NeuronLayer:
    def __init__(self,resistance = 1,voltage_initial=0,voltage_threshold = 1,spike_count = 10,tau = 7,number_neurons = 2,previous_layer = None):
        self.neurons = []
        self.voltages = []
        self.firingRates = []
        self.timestamps = []
        self.totalTime = 0
        for i in range(0,number_neurons):
            new_neuron = Neuron()
            if not previous_layer is None:
                for old_neuron in previous_layer.neurons:
                    new_neuron.connectPreSynaptic(old_neuron)
            self.addNeuron(new_neuron)


    def addNeuron(self,neuron):
        self.neurons.append(neuron)
    
    def step(self,timestep = 1):
        self.totalTime += timestep
        for n in self.neurons:
            n.step(timestep)
    
    def resetTimer(self):
        self.totalTime = 0
        for n in self.neurons:
            n.resetTimer()

    def logInfo(self):
        self.timestamps.append(self.totalTime)
        voltages = []
        firingRates = []
        for n in self.neurons:
            voltages.append(n.voltage)
            firingRates.append(n.getActivity())
        self.voltages.append(voltages)
        self.firingRates.append(firingRates)

    def resetLog(self):
        self.voltages = []
        self.timestamps = []
        self.firingRates = []
        self.totalTime = 0

    def train(self,timestep = 1):
        for n in self.neurons:
            n.train(timestep=timestep)

    def resetEpoch(self):
        for n in self.neurons:
            n.resetEpoch()

if __name__ == "__main__":
    InputLayer = NeuronLayer(number_neurons=0)
    input1 = InputNeuron()
    input2 = InputNeuron()
    InputLayer.addNeuron(input1)
    InputLayer.addNeuron(input2)

    layers = []
    layers.append(NeuronLayer(number_neurons=2))
    layers.append(NeuronLayer(number_neurons=2,previous_layer=layers[0]))
    layers.append(NeuronLayer(number_neurons=2,previous_layer=layers[1])) #output layer

    teacher1 = InputNeuron()
    teacher2 = InputNeuron()
    layers[0].neurons[0].connectPreSynaptic(input1,weight=1)
    layers[0].neurons[1].connectPreSynaptic(input2,weight=1)
    layers[2].neurons[0].connectPreSynaptic(teacher1,weight=1)
    layers[2].neurons[1].connectPreSynaptic(teacher2,weight=1)

    ZERO_CURRENT_INPUT = 17 # map input bits to currents
    ONE_CURRENT_INPUT = 39

    AFFIRMITIVE_INPUT = 60
    input_cases = [
        [ZERO_CURRENT_INPUT,ZERO_CURRENT_INPUT,AFFIRMITIVE_INPUT,0],
        [ZERO_CURRENT_INPUT,ONE_CURRENT_INPUT,0,AFFIRMITIVE_INPUT],
        [ONE_CURRENT_INPUT,ZERO_CURRENT_INPUT,0,AFFIRMITIVE_INPUT],
        [ONE_CURRENT_INPUT,ONE_CURRENT_INPUT,AFFIRMITIVE_INPUT,0]
    ]
    input_output = [
        input1,
        input2,
        teacher1,
        teacher2
    ]
    dt = .01 # timestep in ms
    epochTime = EPOCH_TIME # time for each epoch (between trainings)
    simulationTime = 100 # total time to simulate for

    #initialize izh neuron
    for i in range(0,1000):
        for l in layers:
            l.step(dt)
    for l in layers:
        l.resetLog()
    # TRAIN DATA FIRST
    for num_time in range(0,8):
        for casenum in range(0,len(input_cases)):
            case = input_cases[casenum]
            for i in range(0,4):
                input_output[i].current = case[i]
            for i in range(0,math.floor(simulationTime/epochTime)):
                for i2 in range(0,math.floor(epochTime/dt)):
                    for l in layers:
                        #print(str(len(l.neurons)))
                        l.step(timestep = dt)
                        l.logInfo()
                for num_training in range(0,10):
                    for l in layers:
                        l.train()
                for l in layers:
                    l.resetEpoch()  
            fig = plt.figure()
            for li in range(0,len(layers)):
                plt.subplot(3,1,li+1)
                l = layers[li]
                for i in range(0,len(l.neurons)):
                    plt.plot(l.timestamps,[x[i] for x in l.voltages],label="Layer "+str(li)+" neuron "+str(i),alpha=.5)
                l.resetLog()
                plt.legend()
            #plt.savefig('case'+str(casenum)+'numtime'+str(num_time)+'.png')
            plt.close()
        
    test_cases = [
        [ZERO_CURRENT_INPUT,ZERO_CURRENT_INPUT,0,0],
        [ZERO_CURRENT_INPUT,ONE_CURRENT_INPUT,0,0],
        [ONE_CURRENT_INPUT,ZERO_CURRENT_INPUT,0,0],
        [ONE_CURRENT_INPUT,ONE_CURRENT_INPUT,0,0]
    ]

    simulationTime=100
    for casenum in range(0,len(test_cases)):
        case = test_cases[casenum]
        for i in range(0,4):
            input_output[i].current = case[i]
        for i in range(0,math.floor(simulationTime/epochTime)):
            for i2 in range(0,math.floor(epochTime/dt)):
                for l in layers:
                    #print(str(len(l.neurons)))
                    l.step(timestep = dt)
                    l.logInfo()
            for l in layers:
                l.resetEpoch()  
        fig = plt.figure()
        for li in range(0,len(layers)):
            plt.subplot(3,1,li+1)
            l = layers[li]
            colors = ['r','b','g']
            for i in range(0,len(l.neurons)):
                plt.plot(l.timestamps,[x[i] for x in l.voltages],label="Layer "+str(li)+" neuron "+str(i),alpha=.2,color=colors[i])
            plt.legend()
        print("ZERO:",layers[2].neurons[0].getActivity())
        print("ONE:",layers[2].neurons[1].getActivity())
        if (layers[2].neurons[0].getActivity() > layers[2].neurons[1].getActivity()):
            print("ZERO")
        else:
            print("ONE")
        plt.savefig('testcase'+str(casenum)+'.png')
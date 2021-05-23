"""
Temporary enviornment functional as standin
Sends random states
#"""
__author__ = 'BlackDChase,MR-TLL'
__version__ = '1.3.5'

# Imports
import torch
from lstm import LSTM
output_size = 13
input_dim = output_size
hidden_dim = 128
layer_dim = 1

# Temporary Environment
class TempEnv:
    """
    We don't have much info about environment step,reset method
    How it will cater multiple boss's trajectory
    #"""
    def __init__(self,model,stateSize=13,dataset_path="../Dataset/col13data1.csv",actionSpace=[-15,-10,0,10,15],debug=False):
        self.stateSize = stateSize
        self.actionSpace = actionSpace
        self.state = None
        self.model = model
        self.data_path = dataset_path
        self.timeStep = 100
        self.reset()
        self.debug=debug
        pass

    def step(self,action):
        if action is not None:
            action = int(self.actionSpace[action])
        print(self.state[0][0].shape,self.state[0][0])
        price = int(self.state[0][0]*action)
        nextState = self.model(torch.Tensor(self.previous))
        reward = price*(nextState[0][1]+nextState[0][1])
        nextState[0][0]=price
        self.previous = self.previous[:-self.timeStep+1]
        self.previous[0].append(nextState.tolist())
        info="Step Done"
        nextState.detach()
        self.state = nextState
        return nextState,reward,info

    def reset(self):
        s1 = torch.rand(self.stateSize)
        s2 = torch.rand(self.stateSize)
        s3 = torch.rand(self.stateSize)
        start = torch.cat((s1,s2,s3)).reshape(1,3,self.stateSize)
        state=None
        for i in range(4):
            state=self.model(start)
            state=state.reshape(1,1,self.stateSize)
            torch.cat((start,state),dim=1)
        state=state.detach()
        self.state=state
        self.previous = start.tolist()
        return self.state
    pass

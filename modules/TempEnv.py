"""
Temporary enviornment functional as standin
Sends random states
#"""
__author__ = 'BlackDChase,MR-TLL'
__version__ = '0.1.0'
# Imports
import torch

# Temporary Environment
class TempEnv:
    """
    We don't have much info about environment step,reset method
    How it will cater multiple boss's trajectory
    #"""
    def __init__(self,stateSize,actions=None):
        self.stateSize = stateSize
        self.actions = actions
        self.state = None
        pass

    def step(self,action):
        if self.actions is not None:
            action = self.actions[action]
        nextState = torch.rand(self.stateSize)
        reward = 10
        info="Step Done"
        self.state = nextState
        return nextState,reward,info

    def reset(self):
        self.state = torch.rand(self.stateSize)
        return self.state
    pass

import torch
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
        reward = nextState[0]*(nextState[1]-self.state[1])
        info="Step Done"
        self.state = nextState
        return nextState,reward,info

    def reset(self):
        self.state = torch.rand(self.stateSize)
        return self.state
    pass

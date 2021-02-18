"""
GOD AGENT
- Will take final action
- A supervising psuedo agent that takes action
- Definations required:
    - policy/critic Net
    - state
    - action

"""
__author__ = 'BlackDChase'
__version__ = '0.0.1'

import torch as pt

class GOD:
    def __init__(self,maxEpisode=100,nAgent=1,debug=False,trajectoryLenght=25):
        self.name="GOD"
        self.setMaxEpisode(maxEpisode)
        self.setNumberOfAgent(nAgent)
        self.setTrajectoryLength(trajectoryLenght)
        self.bossAgent = []
        self.price = 0
        # To be defined Later
        '''
        self.policyNet =
        self.criticNet =
        self.state =
        #'''
        pass

    def setNumberOfAgent(self,nAgent):
        self.nAgent = nAgent
        pass

    def setMaxEpisode(self,maxEpisode):
        self.maxEpisode = maxEpisode
        pass

    def setTrajectoryLength(self,trajectoryLenght):
        self.trajectoryLength = trajectoryLenght
        pass

    def getState(self):
        # To be defined Later
        pass

    def takeAction(self):
        # To be defined Later
        pass

    def initateBoss(self):
        # To be defined Later
        pass

    def trainBoss(self):
        # To be defined Later
        pass

    pass

class BOSS(GOD):
    def __init__(self,actorLearningRate=0.01,criticLearningRate=0.01):
        super().__init__()
        self.name='BOSS'
        self.a_lr=actorLearningRate
        self.c_lr=criticLearningRate
    pass

    def train(self):
        # here the main logic of training of A2C will be present
        trajectory_list=GatherAndStore()
        for i in trajectory_list:
            self.v_val_pred+=CalculateV_p()
            self.v_val_target+=CalculateV_tar()
            self.advantage=CalculateGAE()
        CalculateAndUpdateL_P()  # calculate  policy loss and update policy network
        CalculateAndUpdateL_C() # calculate critic loss and update critic network
        pass


    def GatherAndStore(self):
        # gather a trajectory by acting in the enviornment using current policy
        pass

    def CalculateV_p(self):
        # calculate the predictied v value by using critic network
        pass

    def CalculateV_tar(self):
        # calculate the target value v_tar using critic network
        pass

    def CalculateGAE(self):
        # calculate the Advantage uisng the critic network
        pass

    def CalculateAndUpdateL_P(self):
        pass

    def CalculateAndUpdateL_C(self):
        pass

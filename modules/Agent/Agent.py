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
    pass
